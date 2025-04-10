function(enableGpuDirect exename)
    if(GPU_DIRECT)
        target_compile_definitions(${exename} PRIVATE USE_GPU_DIRECT)
    endif()
endfunction()

set(exename volana)
add_executable(${exename} main.cpp)
message(HEFFTE_INC_DIR="${HEFFTE_PATH}/include")
target_include_directories(${exename} PUBLIC ${CSTONE_DIR} ${MPI_CXX_INCLUDE_PATH} ${PROJECT_SOURCE_DIR}/extern/io ${HEFFTE_INC_DIR})
target_link_libraries(${exename} PRIVATE io OpenMP::OpenMP_CXX ${MPI_CXX_LIBARIES} Heffte::Heffte)
install(TARGETS ${exename} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CMAKE_HIP_COMPILER)
    add_executable(${exename}-hip main.cpp)
    target_compile_definitions(${exename}-hip PRIVATE USE_CUDA THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    target_link_libraries(${exename}-hip PRIVATE io OpenMP::OpenMP_CXX
        ${MPI_CXX_LIBRARIES} hip::host)
    set_target_properties(${exename}-hip PROPERTIES LINKER_LANGUAGE CXX)
    enableGpuDirect(${exename}-hip)
    install(TARGETS ${exename}-hip RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
