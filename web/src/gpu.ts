/// <reference types="@webgpu/types" />
/** WebGPU availability checks, device acquisition and small pipeline helpers. */

export class WebGpuUnavailableError extends Error {
  override name = 'WebGpuUnavailableError';
}

export interface WebGpuAvailability {
  readonly available: boolean;
  /** Human-readable reason when unavailable, else an empty string. */
  readonly reason: string;
}

/**
 * Cheap synchronous check: is the WebGPU API exposed here at all?
 * A `true` result can still fail at adapter request time; use
 * {@link checkWebGpu} for a definitive answer.
 */
export function isWebGpuAvailable(): WebGpuAvailability {
  if (typeof navigator === 'undefined') {
    return { available: false, reason: 'no navigator object (not running in a browser)' };
  }
  if (typeof isSecureContext !== 'undefined' && !isSecureContext) {
    return {
      available: false,
      reason:
        'WebGPU requires a secure context (HTTPS or http://localhost); this page was opened over plain HTTP from another host',
    };
  }
  if (!('gpu' in navigator) || navigator.gpu === undefined) {
    return { available: false, reason: 'this browser does not expose WebGPU (navigator.gpu is missing)' };
  }
  return { available: true, reason: '' };
}

/** Asynchronous check that also requests an adapter. */
export async function checkWebGpu(): Promise<WebGpuAvailability> {
  const quick = isWebGpuAvailable();
  if (!quick.available) return quick;
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { available: false, reason: 'no WebGPU adapter is available (GPU blocklisted or disabled)' };
    return { available: true, reason: '' };
  } catch (e) {
    return { available: false, reason: `requesting a WebGPU adapter failed: ${String(e)}` };
  }
}

/** Requests a device with buffer limits large enough for the decoder's activations. */
export async function requestDevice(): Promise<GPUDevice> {
  const quick = isWebGpuAvailable();
  if (!quick.available) throw new WebGpuUnavailableError(quick.reason);
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new WebGpuUnavailableError('no WebGPU adapter is available (GPU blocklisted or disabled)');
  const want = 1 << 30;
  const requiredFeatures: GPUFeatureName[] = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(want, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(want, adapter.limits.maxBufferSize),
    },
  });
  return device;
}

export type BindingKind = 'uniform' | 'uniform-dynamic' | 'read' | 'rw';

export interface Kernel {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  readonly layout: GPUBindGroupLayout;
}

export function makeKernel(
  device: GPUDevice,
  label: string,
  code: string,
  bindings: readonly BindingKind[],
  constants?: Record<string, number>,
): Kernel {
  const layout = device.createBindGroupLayout({
    label,
    entries: bindings.map((kind, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer:
        kind === 'uniform'
          ? { type: 'uniform' as const }
          : kind === 'uniform-dynamic'
            ? { type: 'uniform' as const, hasDynamicOffset: true }
            : kind === 'read'
              ? { type: 'read-only-storage' as const }
              : { type: 'storage' as const },
    })),
  });
  const module = device.createShaderModule({ label, code });
  const compute: GPUProgrammableStage = constants
    ? { module, entryPoint: 'main', constants }
    : { module, entryPoint: 'main' };
  const pipeline = device.createComputePipeline({
    label,
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute,
  });
  return { label, pipeline, layout };
}
