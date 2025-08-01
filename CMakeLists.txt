cmake_minimum_required(VERSION 3.16)
project(QuantXEngine VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Find required packages
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Find ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include/onnxruntime
          /usr/include/onnxruntime
          ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include
)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS /usr/local/lib
          /usr/lib
          ${CMAKE_SOURCE_DIR}/third_party/onnxruntime/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime not found. Please install ONNX Runtime or set the paths manually.")
endif()

# Find nlohmann/json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    # Try to find it as a system package
    find_path(NLOHMANN_JSON_INCLUDE_DIR 
        NAMES nlohmann/json.hpp
        PATHS /usr/local/include
              /usr/include
              ${CMAKE_SOURCE_DIR}/third_party/json/include
    )
    
    if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR "nlohmann/json not found. Please install nlohmann-json-dev or set the path manually.")
    endif()
endif()

# Find WebSocket++
find_path(WEBSOCKETPP_INCLUDE_DIR
    NAMES websocketpp/config/asio_no_tls.hpp
    PATHS /usr/local/include
          /usr/include
          ${CMAKE_SOURCE_DIR}/third_party/websocketpp
)

if(NOT WEBSOCKETPP_INCLUDE_DIR)
    message(FATAL_ERROR "WebSocket++ not found. Please install libwebsocketpp-dev or set the path manually.")
endif()

# Find Boost (required by WebSocket++)
find_package(Boost REQUIRED COMPONENTS system)

# Find OpenSSL (optional, for secure connections)
find_package(OpenSSL)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${WEBSOCKETPP_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

if(nlohmann_json_FOUND)
    # Use the found package
else()
    include_directories(${NLOHMANN_JSON_INCLUDE_DIR})
endif()

# Source files
set(CORE_SOURCES
    src/core/market_data_handler.cpp
)

set(ML_SOURCES
    src/ml/onnx_predictor.cpp
)

set(RISK_SOURCES
    src/risk/risk_manager.cpp
)

set(TRADING_SOURCES
    src/trading/paper_trading.cpp
)

set(MAIN_SOURCES
    src/main.cpp
)

# Create the main executable
add_executable(quantx_engine
    ${MAIN_SOURCES}
    ${CORE_SOURCES}
    ${ML_SOURCES}
    ${RISK_SOURCES}
    ${TRADING_SOURCES}
)

# Link libraries
target_link_libraries(quantx_engine
    ${ONNXRUNTIME_LIB}
    ${Boost_LIBRARIES}
    Threads::Threads
)

if(nlohmann_json_FOUND)
    target_link_libraries(quantx_engine nlohmann_json::nlohmann_json)
endif()

if(OpenSSL_FOUND)
    target_link_libraries(quantx_engine OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(quantx_engine PRIVATE HAVE_OPENSSL)
endif()

# Create a library for testing
add_library(quantx_lib STATIC
    ${CORE_SOURCES}
    ${ML_SOURCES}
    ${RISK_SOURCES}
    ${TRADING_SOURCES}
)

target_link_libraries(quantx_lib
    ${ONNXRUNTIME_LIB}
    ${Boost_LIBRARIES}
    Threads::Threads
)

if(nlohmann_json_FOUND)
    target_link_libraries(quantx_lib nlohmann_json::nlohmann_json)
endif()

# Install targets
install(TARGETS quantx_engine
    RUNTIME DESTINATION bin
)

install(DIRECTORY models/
    DESTINATION share/quantx/models
    FILES_MATCHING PATTERN "*.onnx"
)

# Create a simple test executable
add_executable(test_quantx
    tests/test_main.cpp
)

target_link_libraries(test_quantx quantx_lib)

# Enable testing
enable_testing()
add_test(NAME quantx_tests COMMAND test_quantx)

# Print configuration summary
message(STATUS "QuantX Engine Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "  WebSocket++: ${WEBSOCKETPP_INCLUDE_DIR}")
message(STATUS "  Boost: ${Boost_LIBRARIES}")
if(OpenSSL_FOUND)
    message(STATUS "  OpenSSL: Found")
else()
    message(STATUS "  OpenSSL: Not found (secure connections disabled)")
endif()
