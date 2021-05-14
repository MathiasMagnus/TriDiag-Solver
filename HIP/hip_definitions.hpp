#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_CPU_RT__)
#define cudaAddressModeClamp hipAddressModeClamp
#define cudaBindTexture hipBindTexture
#define cudaChannelFormatDesc hipChannelFormatDesc
#define cudaChannelFormatKindFloat hipChannelFormatKindFloat
#define cudaChannelFormatKindSigned hipChannelFormatKindSigned
#define cudaCreateChannelDesc hipCreateChannelDesc
#define cudaCreateTextureObject hipCreateTextureObject
#define cudaDestroyTextureObject hipDestroyTextureObject
#define cudaDeviceAttributeComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDeviceAttributeComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDeviceAttributeMaxGridDimX hipDeviceAttributeMaxGridDimX
#define cudaDeviceAttributeMultiprocessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceMapHost hipDeviceMapHost
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceProp_t hipDeviceProp_t
#define cudaDeviceReset hipDeviceReset
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorInvalidConfiguration hipErrorInvalidConfiguration
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorNotSupported hipErrorNotSupported
#define cudaErrorOutOfMemory hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreateWithFlags
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFilterModePoint hipFilterModePoint
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaFuncAttributes hipFuncAttributes
#ifndef __HIP_CPU_RT__
#define cudaFuncCachePreferEqual hipFuncCachePreferEqual
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaFuncGetAttributes hipFuncGetAttributes
#define cudaFuncSetCacheConfig(a,b) hipFuncSetCacheConfig(reinterpret_cast<const void*>(a), b)
#else
#define cudaFuncCachePreferEqual
#define cudaFuncCachePreferL1
#define cudaFuncGetAttributes
#define cudaFuncSetCacheConfig(a,b)
#endif
#define cudaFuncSetSharedMemConfig(...)
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostFree hipHostFree
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostMalloc hipHostMalloc
#define cudaHostMallocMapped hipHostMallocMapped
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaLaunchKernelGGL hipLaunchKernelGGL
#define cudaMalloc hipMalloc
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset hipMemset
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaReadModeElementType hipReadModeElementType
#define cudaResourceDesc hipResourceDesc
#define cudaResourceTypeLinear hipResourceTypeLinear
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaSetDevice hipSetDevice
#define cudaSetValidDevices(a,b) hipSuccess
#define cudaSharedMemBankSizeEightByte hipSharedMemBankSizeEightByte
#define cudaSharedMemBankSizeFourByte hipSharedMemBankSizeFourByte
#define cudaSharedMemConfig hipSharedMemConfig
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDefault hipStreamDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaTextureDesc hipTextureDesc
#define cudaTextureObject_t hipTextureObject_t
#define cudaUnbindTexture hipUnbindTexture

#define CU_EVENT_DISABLE_TIMING hipEventDisableTiming

#define cub hipcub

#define cuFloatComplex hipFloatComplex
#define cuComplex hipComplex
#define cuDoubleComplex hipDoubleComplex
#define cuConjf hipConjf
#define cuConj hipConj
#define cuCabsf hipCabsf
#define cuCabs hipCabs
#define cuCaddf hipCaddf
#define cuCadd hipCadd
#define cuCsubf hipCsubf
#define cuCsub hipCsub
#define cuCdivf hipCdivf
#define cuCdiv hipCdiv
#define cuCmulf hipCmulf
#define cuCmul hipCmul
#define make_cuComplex hipComplex
#define make_cuDoubleComplex hipDoubleComplex
#define cuCrealf hipCrealf
#define cuCimagf hipCimagf
#define cuCreal hipCreal
#define cuCimag hipCimag
#define cuFma hipFma
#define cuCfmaf hipCfmaf
#define cuCfma hipCfma

#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_R2C HIPFFT_R2C
#define CUFFT_C2R HIPFFT_C2R
#define cufft hipfft
#define cufftComplex hipfftComplex
#define cufftDestroy hipfftDestroy
#define cufftDoubleComplex hipfftDoubleComplex
#define cufftExecC2R hipfftExecC2R
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecR2C hipfftExecR2C
#define cufftExecZ2D hipfftExecZ2D
#define cufftHandle hipfftHandle
#define cufftPlan3d hipfftPlan3d

#define CURAND_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#define curandCreateGenerator(a,b) hiprandCreateGenerator(a,b)
#define curandGenerateNormalDouble(a,b,c,d,e) hiprandGenerateNormalDouble(a,b,c,d,e)
#define curandGenerator_t hiprandGenerator_t
#define curandSetPseudoRandomGeneratorSeed(a,b) hiprandSetPseudoRandomGeneratorSeed(a,b)

#endif
