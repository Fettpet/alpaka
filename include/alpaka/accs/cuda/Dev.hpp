/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/accs/cuda/Common.hpp>

#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType
#include <alpaka/traits/Wait.hpp>           // CurrentThreadWaitFor

#include <iostream>                         // std::cout
#include <sstream>                          // std::stringstream
#include <limits>                           // std::numeric_limits
#include <stdexcept>                        // std::runtime_error

namespace alpaka
{
    namespace accs
    {
        namespace cuda
        {
            namespace detail
            {
                class AccCuda;
                class DevManCuda;

                //#############################################################################
                //! The CUDA accelerator device handle.
                //#############################################################################
                class DevCuda
                {
                    friend class DevManCuda;
                protected:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCuda() = default;
                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCuda(DevCuda const &) = default;
#if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCuda(DevCuda &&) = default;
#endif
                    //-----------------------------------------------------------------------------
                    //! Assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(DevCuda const &) -> DevCuda & = default;
                    //-----------------------------------------------------------------------------
                    //! Equality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator==(DevCuda const & rhs) const
                    -> bool
                    {
                        return m_iDevice == rhs.m_iDevice;
                    }
                    //-----------------------------------------------------------------------------
                    //! Inequality comparison operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator!=(DevCuda const & rhs) const
                    -> bool
                    {
                        return !((*this) == rhs);
                    }

                public:
                    int m_iDevice;
                };

                //#############################################################################
                //! The CUDA accelerator device manager.
                //#############################################################################
                class DevManCuda
                {
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevManCuda() = delete;

                    //-----------------------------------------------------------------------------
                    //! \return The number of devices available.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getDevCount()
                    -> std::size_t
                    {
                        int iNumDevices(0);
                        ALPAKA_CUDA_RT_CHECK(cudaGetDeviceCount(&iNumDevices));

                        return static_cast<std::size_t>(iNumDevices);
                    }
                    //-----------------------------------------------------------------------------
                    //! \return The number of devices available.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getDevByIdx(
                        std::size_t const & uiIdx)
                    -> DevCuda
                    {
                        DevCuda device;

                        std::size_t const uiNumDevices(getDevCount());
                        if(uiIdx < uiNumDevices)
                        {
                            device.m_iDevice = static_cast<int>(uiIdx);
                        }
                        else
                        {
                            std::stringstream ssErr;
                            ssErr << "Unable to return device handle for device " << uiIdx << " because there are only " << uiNumDevices << " CUDA devices!";
                            throw std::runtime_error(ssErr.str());
                        }

                        return device;
                    }
/*                private:
                    //-----------------------------------------------------------------------------
                    //! \return The handle to the currently used device.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto getCurrentDev()
                    -> DevCuda
                    {
                        DevCuda dev;
                        ALPAKA_CUDA_RT_CHECK(cudaGetDevice(&dev.m_iDevice));
                        return dev;
                    }
                    //-----------------------------------------------------------------------------
                    //! Sets the device to use with this accelerator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST static auto setCurrentDev(
                        DevCuda const & dev)
                    -> void
                    {
                        ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
#if 0
                        std::size_t uiNumDevices(getDevCount());
                        if(uiNumDevices < 1)
                        {
                            throw std::runtime_error("No CUDA capable devices detected!");
                        }
                        else if(uiNumDevices < dev.m_iDevice)
                        {
                            std::stringstream ssErr;
                            ssErr << "No CUDA device " << dev.m_iDevice << " available, only " << uiNumDevices << " devices found!";
                            throw std::runtime_error(ssErr.str());
                        }

                        cudaDeviceProp devProp;
                        ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(&devProp, dev.m_iDevice));
                        // Default compute mode (Multiple threads can use cudaSetDevice() with this device)
                        if(devProp.computeMode == cudaComputeModeDefault)
                        {
                            ALPAKA_CUDA_RT_CHECK(cudaSetDevice(dev.m_iDevice));
                            std::cout << "Set device to " << dev.m_iDevice << ": " << std::endl;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDeviceProperties(devProp);
#else
                            std::cout << devProp.name << std::endl;
#endif
                        }
                        // Compute-exclusive-thread mode (Only one thread in one process will be able to use cudaSetDevice() with this device)
                        else if(devProp.computeMode == cudaComputeModeExclusive)
                        {
                            std::cout << "Requested device is in computeMode cudaComputeModeExclusive.";
                            // \TODO: Are we allowed to use the device in compute mode cudaComputeModeExclusive?
                        }
                        // Compute-prohibited mode (No threads can use cudaSetDevice() with this device)
                        else if(devProp.computeMode == cudaComputeModeProhibited)
                        {
                            std::cout << "Requested device is in computeMode cudaComputeModeProhibited. It can not be selected!";
                        }
                        // Compute-exclusive-process mode (Many threads in one process will be able to use cudaSetDevice() with this device)
                        else if(devProp.computeMode == cudaComputeModeExclusiveProcess)
                        {
                            std::cerr << "Requested device is in computeMode cudaComputeModeExclusiveProcess.";
                            // \TODO: Are we allowed to use the device in compute mode cudaComputeModeExclusiveProcess?
                        }
                        else
                        {
                            std::cerr << "unknown computeMode!";
                        }

                        // Instruct CUDA to actively spin when waiting for results from the device.
                        // This can decrease latency when waiting for the device, but may lower the performance of CPU threads if they are performing work in parallel with the CUDA thread.
                        ALPAKA_CUDA_RT_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
#else
                        ALPAKA_CUDA_RT_CHECK(cudaSetDevice(dev.m_iDevice));
#endif
                    }*/

                private:
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                    //-----------------------------------------------------------------------------
                    //! Prints all the device properties to std::cout.
                    //-----------------------------------------------------------------------------
                    static auto printDeviceProperties(
                        cudaDeviceProp const & devProp)
                    -> void
                    {
                        std::size_t const uiKiB(1024);
                        std::size_t const uiMiB(uiKiB * uiKiB);
                        std::cout << "name: " << devProp.name << std::endl;
                        std::cout << "totalGlobalMem: " << devProp.totalGlobalMem/uiMiB << " MiB" << std::endl;
                        std::cout << "sharedMemPerBlock: " << devProp.sharedMemPerBlock/uiKiB << " KiB" << std::endl;
                        std::cout << "regsPerBlock: " << devProp.regsPerBlock << std::endl;
                        std::cout << "warpSize: " << devProp.warpSize << std::endl;
                        std::cout << "memPitch: " << devProp.memPitch << " B" << std::endl;
                        std::cout << "maxThreadsPerBlock: " << devProp.maxThreadsPerBlock << std::endl;
                        std::cout << "maxThreadsDim[3]: (" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")" << std::endl;
                        std::cout << "maxGridSize[3]: (" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")" << std::endl;
                        std::cout << "clockRate: " << devProp.clockRate << " kHz" << std::endl;
                        std::cout << "totalConstMem: " << devProp.totalConstMem/uiKiB << " KiB" << std::endl;
                        std::cout << "major: " << devProp.major << std::endl;
                        std::cout << "minor: " << devProp.minor << std::endl;
                        std::cout << "textureAlignment: " << devProp.textureAlignment << std::endl;
                        std::cout << "texturePitchAlignment: " << devProp.texturePitchAlignment << std::endl;
                        //std::cout << "deviceOverlap: " << devProp.deviceOverlap << std::endl;    // Deprecated
                        std::cout << "multiProcessorCount: " << devProp.multiProcessorCount << std::endl;
                        std::cout << "kernelExecTimeoutEnabled: " << devProp.kernelExecTimeoutEnabled << std::endl;
                        std::cout << "integrated: " << devProp.integrated << std::endl;
                        std::cout << "canMapHostMemory: " << devProp.canMapHostMemory << std::endl;
                        std::cout << "computeMode: " << devProp.computeMode << std::endl;
                        std::cout << "maxTexture1D: " << devProp.maxTexture1D << std::endl;
                        std::cout << "maxTexture1DLinear: " << devProp.maxTexture1DLinear << std::endl;
                        std::cout << "maxTexture2D[2]: " << devProp.maxTexture2D[0] << "x" << devProp.maxTexture2D[1] << std::endl;
                        std::cout << "maxTexture2DLinear[3]: " << devProp.maxTexture2DLinear[0] << "x" << devProp.maxTexture2DLinear[1] << "x" << devProp.maxTexture2DLinear[2] << std::endl;
                        std::cout << "maxTexture2DGather[2]: " << devProp.maxTexture2DGather[0] << "x" << devProp.maxTexture2DGather[1] << std::endl;
                        std::cout << "maxTexture3D[3]: " << devProp.maxTexture3D[0] << "x" << devProp.maxTexture3D[1] << "x" << devProp.maxTexture3D[2] << std::endl;
                        std::cout << "maxTextureCubemap: " << devProp.maxTextureCubemap << std::endl;
                        std::cout << "maxTexture1DLayered[2]: " << devProp.maxTexture1DLayered[0] << "x" << devProp.maxTexture1DLayered[1] << std::endl;
                        std::cout << "maxTexture2DLayered[3]: " << devProp.maxTexture2DLayered[0] << "x" << devProp.maxTexture2DLayered[1] << "x" << devProp.maxTexture2DLayered[2] << std::endl;
                        std::cout << "maxTextureCubemapLayered[2]: " << devProp.maxTextureCubemapLayered[0] << "x" << devProp.maxTextureCubemapLayered[1] << std::endl;
                        std::cout << "maxSurface1D: " << devProp.maxSurface1D << std::endl;
                        std::cout << "maxSurface2D[2]: " << devProp.maxSurface2D[0] << "x" << devProp.maxSurface2D[1] << std::endl;
                        std::cout << "maxSurface3D[3]: " << devProp.maxSurface3D[0] << "x" << devProp.maxSurface3D[1] << "x" << devProp.maxSurface3D[2] << std::endl;
                        std::cout << "maxSurface1DLayered[2]: " << devProp.maxSurface1DLayered[0] << "x" << devProp.maxSurface1DLayered[1] << std::endl;
                        std::cout << "maxSurface2DLayered[3]: " << devProp.maxSurface2DLayered[0] << "x" << devProp.maxSurface2DLayered[1] << "x" << devProp.maxSurface2DLayered[2] << std::endl;
                        std::cout << "maxSurfaceCubemap: " << devProp.maxSurfaceCubemap << std::endl;
                        std::cout << "maxSurfaceCubemapLayered[2]: " << devProp.maxSurfaceCubemapLayered[0] << "x" << devProp.maxSurfaceCubemapLayered[1] << std::endl;
                        std::cout << "surfaceAlignment: " << devProp.surfaceAlignment << std::endl;
                        std::cout << "concurrentKernels: " << devProp.concurrentKernels << std::endl;
                        std::cout << "ECCEnabled: " << devProp.ECCEnabled << std::endl;
                        std::cout << "pciBusID: " << devProp.pciBusID << std::endl;
                        std::cout << "pciDeviceID: " << devProp.pciDeviceID << std::endl;
                        std::cout << "pciDomainID: " << devProp.pciDomainID << std::endl;
                        std::cout << "tccDriver: " << devProp.tccDriver << std::endl;
                        std::cout << "asyncEngineCount: " << devProp.asyncEngineCount << std::endl;
                        std::cout << "unifiedAddressing: " << devProp.unifiedAddressing << std::endl;
                        std::cout << "memoryClockRate: " << devProp.memoryClockRate << " kHz" << std::endl;
                        std::cout << "memoryBusWidth: " << devProp.memoryBusWidth << " b" << std::endl;
                        std::cout << "l2CacheSize: " << devProp.l2CacheSize << " B" << std::endl;
                        std::cout << "maxThreadsPerMultiProcessor: " << devProp.maxThreadsPerMultiProcessor << std::endl;
                    }
#endif
                };

                class StreamCuda;
            }
        }
    }

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The CUDA accelerator device accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::cuda::detail::DevCuda>
            {
                using type = accs::cuda::detail::AccCuda;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The CUDA accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::DevCuda;
            };
            //#############################################################################
            //! The CUDA accelerator device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::cuda::detail::DevManCuda>
            {
                using type = accs::cuda::detail::DevCuda;
            };

            //#############################################################################
            //! The CUDA accelerator device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                accs::cuda::detail::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    accs::cuda::detail::DevCuda const & dev)
                -> accs::cuda::detail::DevCuda
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The CUDA accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetProps<
                accs::cuda::detail::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getProps(
                    accs::cuda::detail::DevCuda const & dev)
                -> alpaka::dev::DevProps
                {
                    cudaDeviceProp cudaDevProp;
                    ALPAKA_CUDA_RT_CHECK(cudaGetDeviceProperties(
                        &cudaDevProp, 
                        dev.m_iDevice));

                    return alpaka::dev::DevProps(
                        // m_sName
                        cudaDevProp.name,
                        // m_uiMultiProcessorCount
                        static_cast<UInt>(cudaDevProp.multiProcessorCount),
                        // m_uiBlockThreadsCountMax
                        static_cast<UInt>(cudaDevProp.maxThreadsPerBlock),
                        // m_v3uiBlockThreadExtentsMax
                        Vec<3u>(
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[0]),
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[1]),
                            static_cast<UInt>(cudaDevProp.maxThreadsDim[2])),
                        // m_v3uiGridBlockExtentsMax
                        Vec<3u>(
                            static_cast<UInt>(cudaDevProp.maxGridSize[0]),
                            static_cast<UInt>(cudaDevProp.maxGridSize[1]),
                            static_cast<UInt>(cudaDevProp.maxGridSize[2])),
                        // m_uiGlobalMemSizeBytes
                        static_cast<std::size_t>(cudaDevProp.totalGlobalMem));
                        //devProps.m_uiMaxClockFrequencyHz = cudaDevProp.clockRate * 1000;
                }
            };

            //#############################################################################
            //! The CUDA accelerator device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                accs::cuda::detail::DevCuda>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    accs::cuda::detail::DevCuda const & dev)
                -> std::size_t
                {
                    std::size_t freeInternal(0u);
                    std::size_t totalInternal(0u);

                    ALPAKA_CUDA_RT_CHECK(cudaMemGetInfo(
                        &freeInternal, 
                        &totalInternal));

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA accelerator device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                accs::cuda::detail::DevCuda>
            {
                ALPAKA_FCT_HOST static auto reset(
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceReset());
                }
            };

            //#############################################################################
            //! The CUDA accelerator device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::cuda::detail::AccCuda>
            {
                using type = accs::cuda::detail::DevManCuda;
            };
            //#############################################################################
            //! The CUDA accelerator device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::cuda::detail::DevCuda>
            {
                using type = accs::cuda::detail::DevManCuda;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The CUDA accelerator device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::cuda::detail::DevCuda>
            {
                using type = accs::cuda::detail::StreamCuda;
            };
        }

        namespace wait
        {
            //#############################################################################
            //! The CUDA accelerator thread device wait specialization.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                accs::cuda::detail::DevCuda>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    accs::cuda::detail::DevCuda const & dev)
                -> void
                {
                    // \TODO: This should be secured by a lock.

                    // Set the current device to wait for.
                    ALPAKA_CUDA_RT_CHECK(cudaSetDevice(
                        dev.m_iDevice));
                    ALPAKA_CUDA_RT_CHECK(cudaDeviceSynchronize());
                }
            };
        }
    }
}