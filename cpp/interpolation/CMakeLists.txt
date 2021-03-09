set(target_name "interpolation")

add_executable(${target_name}
    main.cpp
    interp.cpp
    interp.hpp
    nedi.cpp
    nedi.hpp
)

target_include_directories(${target_name} PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${HEIF_INCLUDE_DIRS}
)

target_link_libraries(${target_name} PRIVATE 
    ${OpenCV_LIBS}
    ${HEIF_LINK_LIBRARIES}
)