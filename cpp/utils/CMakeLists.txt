set(target_name "utils")

find_package(PkgConfig REQUIRED)
pkg_search_module(HEIF REQUIRED libheif)

add_library(${target_name} STATIC
    heif_reader.cpp
    heif_reader.hpp
)

target_include_directories(${target_name} 
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${OpenCV_INCLUDE_DIRS}
    PRIVATE ${HEIF_INCLUDE_DIRS}
)

target_link_libraries(${target_name} 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE ${HEIF_LINK_LIBRARIES}
)