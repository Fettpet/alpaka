/**
* \file
* Copyright 2016 Benjamin Worpitz
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

#include <alpaka/block/shared/dyn/Traits.hpp>   // AllocVar

#include <alpaka/core/Common.hpp>               // ALPAKA_FN_ACC_NO_CUDA

#include <boost/align.hpp>                      // boost::aligned_alloc

#include <vector>                               // std::vector
#include <memory>                               // std::unique_ptr

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The block shared dynamic memory allocator using the memory given at construction time.
                //#############################################################################
                class BlockSharedMemDynRef
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynRef;

                    //-----------------------------------------------------------------------------
                    //! Default constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynRef(
                        uint8_t * const pBlockSharedMemDyn) :
                            m_pBlockSharedMemDyn(pBlockSharedMemDyn)
                    {}
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynRef(BlockSharedMemDynRef const &) = delete;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA BlockSharedMemDynRef(BlockSharedMemDynRef &&) = delete;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynRef const &) -> BlockSharedMemDynRef & = delete;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA auto operator=(BlockSharedMemDynRef &&) -> BlockSharedMemDynRef & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_ACC_NO_CUDA /*virtual*/ ~BlockSharedMemDynRef() = default;

                public:
                    uint8_t * const m_pBlockSharedMemDyn;
                };

                namespace traits
                {
                    //#############################################################################
                    //!
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynRef>
                    {
                        //-----------------------------------------------------------------------------
                        //
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_ACC_NO_CUDA static auto getMem(
                            block::shared::dyn::BlockSharedMemDynRef const & blockSharedMemDyn)
                        -> T *
                        {
                            return reinterpret_cast<T*>(blockSharedMemDyn.m_pBlockSharedMemDyn);
                        }
                    };
                }
            }
        }
    }
}
