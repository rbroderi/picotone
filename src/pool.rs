//! Recycles the large activation buffers of the networks.
//!
//! Every layer writes its output into a fresh buffer of up to a few
//! megabytes. Handing those back to the allocator and asking for new ones
//! means fresh, unmapped memory each time, and the page faults of first
//! touching it cost more than some of the layers. Buffers are kept here
//! instead and handed out again without being cleared. A request takes the
//! smallest kept buffer that is large enough, however much larger: a few
//! large buffers then serve every layer, and stay in the caches.

#![expect(
    clippy::redundant_pub_crate,
    reason = "the module is private, and rustc's unreachable_pub asks for pub(crate)"
)]

use std::sync::Mutex;

/// Buffers shorter than this are cheap enough to allocate.
const MIN_LEN: usize = 16 * 1024;
/// Buffers longer than this are not kept.
const MAX_LEN: usize = 16 * 1024 * 1024;
/// Buffers kept per worker thread: what one network holds at a time.
const KEPT_PER_THREAD: usize = 4;
/// Buffers kept on top of those.
const KEPT_EXTRA: usize = 2;

static FREE: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());

/// A buffer of `len` values whose contents are unspecified: the caller must
/// write every value before reading it.
pub(crate) fn take(len: usize) -> Vec<f32> {
    let kept = if len < MIN_LEN {
        None
    } else {
        FREE.lock().ok().and_then(|mut free| {
            let best = free
                .iter()
                .enumerate()
                .filter(|(_, buffer)| buffer.capacity() >= len)
                .min_by_key(|(_, buffer)| buffer.capacity())
                .map(|(index, _)| index)?;
            Some(free.swap_remove(best))
        })
    };
    let Some(mut buffer) = kept else {
        return vec![0.0; len];
    };
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    } else {
        buffer.truncate(len);
    }
    buffer
}

/// Hands a buffer back for reuse.
pub(crate) fn give(buffer: Vec<f32>) {
    if !(MIN_LEN..=MAX_LEN).contains(&buffer.capacity()) {
        return;
    }
    let Ok(mut free) = FREE.lock() else {
        return;
    };
    free.push(buffer);
    if free.len() > KEPT_PER_THREAD * rayon::current_num_threads().max(1) + KEPT_EXTRA {
        // The smallest buffer serves the fewest requests.
        let smallest = free
            .iter()
            .enumerate()
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map_or(0, |(index, _)| index);
        drop(free.swap_remove(smallest));
    }
}
