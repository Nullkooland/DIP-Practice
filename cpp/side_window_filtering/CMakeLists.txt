set(target_name "side_window_filtering")

add_executable(${target_name}
    main.cpp
    side_window_filter.cpp
    side_window_filter.hpp
)

target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${target_name} PRIVATE ${OpenCV_LIBS})