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

#include <alpaka/cuda/Common.hpp>
#include <alpaka/cuda/AccCudaFwd.hpp>   // AccCuda

#include <alpaka/interfaces/stream.hpp>

namespace alpaka
{
    namespace stream
    {
        //#############################################################################
        //! The CUDA accelerator stream.
        //#############################################################################
        template<>
        class Stream<AccCuda>
        {
        public:
            using Acc = AccCuda;

        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Stream()
            {
                ALPAKA_CUDA_CHECK(cudaStreamCreate(
                    &m_cudaStream);
            }
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Stream(Stream const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Stream(Stream &&) = default;
            //-----------------------------------------------------------------------------
            //! Assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST Stream & operator=(Stream const &) = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST bool operator==(Stream const & rhs) const
            {
                return m_cudaStream == rhs.m_cudaStream;
            }
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST bool operator!=(Stream const & rhs) const
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST virtual ~Stream() noexcept
            {
                ALPAKA_CUDA_CHECK(cudaStreamDestroy(m_cudaStream));
            }

            cudaStream_t m_cudaStream;
        };

        namespace detail
        {
            //#############################################################################
            //! The CUDA accelerator thread stream waiter.
            //#############################################################################
            template<>
            struct ThreadWaitStream<
                Stream<AccCuda>>
            {
                ALPAKA_FCT_HOST ThreadWaitStream(Stream<AccCuda> const & stream)
                {
                    ALPAKA_CUDA_CHECK(cudaStreamSynchronize(
                        stream.m_cudaStream));
                }
            };

            //#############################################################################
            //! The CUDA accelerator stream event waiter.
            //#############################################################################
            template<>
            struct StreamWaitEvent<
                Stream<AccCuda>,
                event::Event<AccCuda>>
            {
                ALPAKA_FCT_HOST StreamWaitEvent(Stream<AccCuda> const & stream, event::Event<AccCuda> const & event)
                {
                    ALPAKA_CUDA_CHECK(cudaStreamWaitEvent(
                        stream.m_cudaStream,
                        event.m_cudaEvent,
                        0));
                }
            };

            //#############################################################################
            //! The CUDA accelerator stream tester.
            //#############################################################################
            template<>
            struct StreamTest<
                event::Event<AccCuda>>
            {
                ALPAKA_FCT_HOST StreamTest(Stream<AccCuda> const & stream, bool & bTest)
                {
                    auto const ret(cudaStreamQuery(stream.m_cudaStream));
                    if(ret == cudaSuccess)
                    {
                        bTest = true;
                    }
                    else if(ret == cudaErrorNotReady)
                    {
                        bTest = false;
                    }
                    else
                    {
                        throw std::runtime_error(("Unexpected return value '" + std::string(cudaGetErrorString(ret)) + "'from cudaStreamQuery!").c_str());
                    }
                }
            };
        }
    }
}
