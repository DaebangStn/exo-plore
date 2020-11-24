find_package(Boost REQUIRED COMPONENTS program_options)

add_library(util2 SHARED
        log.cpp
        path.cpp
        struct.cpp
)
target_link_libraries(util2 PUBLIC
        Boost::program_options
        ${TinyXML2_LIBRARIES}
        ${TORCH_LIBRARIES}
)
target_include_directories(util2 PRIVATE
        ${Boost_INCLUDE_DIRS}
        ${TinyXML2_INCLUDE_DIRS}
)
target_include_directories(util2 PUBLIC
        ${TORCH_INCLUDE_DIRS}
)