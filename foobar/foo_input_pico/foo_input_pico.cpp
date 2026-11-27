// foo_input_pico: plays picotone's .pico files in foobar2000, HQ and EnCodec,
// with an HQ file's .picol enhancement layer when one lies next to it.
//
// The decoding is picotone's Rust player (picotone-ffi, linked statically; C
// interface in picotone.h). It finds the weights and cx2 tables in, in order:
// the PICOTONE_WEIGHTS / PICOTONE_TABLES environment variables, then
// `weights` and `models` next to this DLL, then one folder up (the x64 DLL of
// a component package sits in an `x64` subfolder).
//
// LGPL-3.0-or-later (foobar/COPYING.LESSER, with the GPL in LICENSE).

#include <SDK/foobar2000.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "picotone.h"

DECLARE_COMPONENT_VERSION(
    "picotone decoder", "0.1.0",
    "Plays picotone .pico files (HQ and EnCodec), with an HQ file's .picol "
    "enhancement layer when it lies next to the file.\n\n"
    "LGPL-3.0-or-later. Weights: Meta's EnCodec (MIT) and picotone's HQ "
    "decoder, derived from Descript's DAC (MIT); see the notices next to the "
    "weights.");
VALIDATE_COMPONENT_FILENAME("foo_input_pico.dll");
DECLARE_FILE_TYPE("picotone files", "*.PICO");

namespace {

// Units (about a second of audio each) per decode_run call.
constexpr uint32_t kUnitsPerRun = 1;

pfc::string8 last_error() { return pfc::string8(pt_last_error()); }

bool is_dir(const std::wstring& path) {
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY);
}

std::wstring env(const wchar_t* name) {
    wchar_t* value = nullptr;
    size_t length = 0;
    if (_wdupenv_s(&value, &length, name) != 0 || value == nullptr) return {};
    std::wstring out(value);
    free(value);
    return out;
}

std::string to_utf8(const std::wstring& text) {
    if (text.empty()) return {};
    const int size = WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string out(size > 0 ? size - 1 : 0, '\0');
    if (size > 0) WideCharToMultiByte(CP_UTF8, 0, text.c_str(), -1, out.data(), size, nullptr, nullptr);
    return out;
}

// The folder holding this DLL.
std::wstring module_dir() {
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&module_dir), &module);
    std::wstring path(MAX_PATH, L'\0');
    for (;;) {
        const DWORD got = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
        if (got < path.size()) {
            path.resize(got);
            break;
        }
        path.resize(path.size() * 2);
    }
    const size_t slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring() : path.substr(0, slash);
}

// Loads the tables once per process; throws when the data is not found.
void ensure_engine() {
    static std::once_flag once;
    static std::string error;
    std::call_once(once, [] {
        std::wstring weights = env(L"PICOTONE_WEIGHTS");
        std::wstring tables = env(L"PICOTONE_TABLES");
        const std::wstring here = module_dir();
        for (const std::wstring& base : {here, here + L"\\.."}) {
            if (weights.empty() && is_dir(base + L"\\weights")) weights = base + L"\\weights";
            if (tables.empty() && is_dir(base + L"\\models")) tables = base + L"\\models";
        }
        if (weights.empty() || tables.empty()) {
            error = "picotone: the weights or models folder was not found next to foo_input_pico.dll "
                    "(or set PICOTONE_WEIGHTS and PICOTONE_TABLES)";
            return;
        }
        if (pt_init(to_utf8(weights).c_str(), to_utf8(tables).c_str()) != 0) {
            error = std::string("picotone: ") + pt_last_error();
        }
    });
    if (!error.empty()) throw exception_io_data(error.c_str());
}

// The .picol next to a .pico: the same path with the extension replaced.
pfc::string8 layer_path(const char* path) {
    pfc::string8 out(path);
    const t_size dot = out.find_last('.');
    const t_size slash = pfc::max_t<t_size>(out.find_last('\\') + 1, out.find_last('/') + 1);
    if (dot != SIZE_MAX && (slash == 0 || dot > slash)) out.truncate(dot);
    out += ".picol";
    return out;
}

const char* layout_name(uint32_t layout) {
    switch (layout) {
        case 0: return "mono";
        case 1: return "left/right";
        case 2: return "mid/side";
        case 3: return "parametric stereo";
        default: return "";
    }
}

class input_pico : public input_stubs {
public:
    ~input_pico() { pt_close(m_stream); }

    void open(service_ptr_t<file> p_filehint, const char* p_path, t_input_open_reason p_reason,
              abort_callback& p_abort) {
        if (p_reason == input_open_info_write) throw exception_tagging_unsupported();
        m_file = p_filehint;
        input_open_file_helper(m_file, p_path, p_reason, p_abort);
        m_stats = m_file->get_stats2_(stats2_all, p_abort);
        m_file->read_till_eof(m_bytes, p_abort);

        // The layer is optional: any trouble reading it leaves the file alone.
        const pfc::string8 layer = layer_path(p_path);
        try {
            if (filesystem::g_exists(layer, p_abort)) {
                service_ptr_t<file> layer_file;
                filesystem::g_open_read(layer_file, layer, p_abort);
                layer_file->read_till_eof(m_layer, p_abort);
            }
        } catch (const exception_aborted&) {
            throw;
        } catch (const std::exception&) {
            m_layer.set_size(0);
        }

        if (pt_header(m_bytes.get_ptr(), m_bytes.get_size(), m_layer.get_ptr(), m_layer.get_size(),
                      &m_header) != 0) {
            throw exception_io_data(last_error());
        }
        if (m_layer.get_size() > 0 && !m_header.layered) {
            if (p_reason == input_open_decode) {
                console::formatter line;
                line << "picotone: " << layer << " does not belong to this file; playing without it";
            }
            m_layer.set_size(0);
        }
    }

    void get_info(file_info& p_info, abort_callback&) {
        const double seconds = static_cast<double>(m_header.length) / m_header.sample_rate;
        p_info.set_length(seconds);
        p_info.info_set_int("samplerate", m_header.sample_rate);
        p_info.info_set_int("channels", m_header.channels);
        p_info.info_set("encoding", "lossy");
        const bool hq = m_header.codec == PT_CODEC_HQ;
        p_info.info_set("codec", hq ? "picotone HQ" : "picotone EnCodec");
        pfc::string8 profile;
        if (hq) {
            profile << "q" << m_header.codebooks;
            if (m_header.layout == 2 && m_header.side_codebooks != m_header.codebooks)
                profile << "/" << m_header.side_codebooks;
            if (m_header.channels > 1) profile << " " << layout_name(m_header.layout);
            if (m_header.layered) profile << " (with .picol)";
        } else {
            profile << (m_header.codec == PT_CODEC_ENCODEC_48K ? "48 kHz" : "24 kHz") << ", "
                    << m_header.codebooks << " codebooks";
        }
        p_info.info_set("codec_profile", profile);
        if (m_header.looped) p_info.info_set("picotone_loop", "yes");
        if (seconds > 0) {
            const double bits = 8.0 * static_cast<double>(m_bytes.get_size() + m_layer.get_size());
            const auto kbps = static_cast<t_int64>(bits / seconds / 1000.0 + 0.5);
            if (kbps > 0) p_info.info_set_bitrate(kbps);
        }

        const ptrdiff_t need = pt_tags(m_bytes.get_ptr(), m_bytes.get_size(), nullptr, 0);
        if (need > 0) {
            std::vector<char> tags(static_cast<size_t>(need));
            if (pt_tags(m_bytes.get_ptr(), m_bytes.get_size(), tags.data(), tags.size()) == need) {
                for (const char* tag = tags.data(); *tag != '\0'; tag += strlen(tag) + 1) {
                    const char* equals = strchr(tag, '=');
                    if (equals == nullptr || equals == tag) continue;
                    p_info.meta_add_ex(tag, equals - tag, equals + 1, SIZE_MAX);
                }
            }
        }
    }

    t_filestats2 get_stats2(uint32_t, abort_callback&) { return m_stats; }
    t_filestats get_file_stats(abort_callback&) { return m_stats.to_legacy(); }

    void decode_initialize(unsigned, abort_callback&) { reopen(0); }

    bool decode_run(audio_chunk& p_chunk, abort_callback& p_abort) {
        p_abort.check();
        if (m_stream == nullptr) return false;
        const float* data = nullptr;
        size_t frames = 0;
        const int32_t got = pt_decode(m_stream, kUnitsPerRun, &data, &frames);
        if (got < 0) throw exception_io_data(last_error());
        if (got == 0) return false;
        p_chunk.set_data_32(data, frames, m_header.channels, m_header.sample_rate);
        return true;
    }

    void decode_seek(double p_seconds, abort_callback& p_abort) {
        p_abort.check();
        const uint64_t sample = audio_math::time_to_samples(p_seconds, m_header.sample_rate);
        if (sample >= m_header.length) {
            pt_close(m_stream);
            m_stream = nullptr;
            return;
        }
        reopen(sample);
    }

    bool decode_can_seek() { return true; }

    void retag(const file_info&, abort_callback&) { throw exception_tagging_unsupported(); }
    void remove_tags(abort_callback&) { throw exception_tagging_unsupported(); }

    static bool g_is_our_content_type(const char*) { return false; }
    static bool g_is_our_path(const char*, const char* p_extension) {
        return stricmp_utf8(p_extension, "pico") == 0;
    }
    static const char* g_get_name() { return "picotone decoder"; }
    static const GUID g_get_guid() {
        static const GUID guid = {0x5b0e2f61, 0x7c3a, 0x4d8e, {0x9a, 0x41, 0x2e, 0x6f, 0x13, 0xc8, 0x77, 0x0d}};
        return guid;
    }

private:
    // A fresh stream from playable sample `sample` (a seek is a reopen); with
    // the layer when it fits, else the file alone.
    void reopen(uint64_t sample) {
        ensure_engine();
        pt_close(m_stream);
        m_stream = nullptr;
        if (m_layer.get_size() > 0) {
            m_stream = pt_open(m_bytes.get_ptr(), m_bytes.get_size(), m_layer.get_ptr(), m_layer.get_size(),
                               sample);
            if (m_stream == nullptr) {
                console::formatter line;
                line << "picotone: ignoring the .picol layer: " << pt_last_error();
                m_layer.set_size(0);
                m_header.layered = 0;
            }
        }
        if (m_stream == nullptr) {
            m_stream = pt_open(m_bytes.get_ptr(), m_bytes.get_size(), nullptr, 0, sample);
        }
        if (m_stream == nullptr) throw exception_io_data(last_error());
    }

    service_ptr_t<file> m_file;
    t_filestats2 m_stats;
    pfc::array_t<uint8_t> m_bytes;
    pfc::array_t<uint8_t> m_layer;
    PtHeader m_header{};
    PtStream* m_stream = nullptr;
};

static input_singletrack_factory_t<input_pico> g_input_pico_factory;

}  // namespace
