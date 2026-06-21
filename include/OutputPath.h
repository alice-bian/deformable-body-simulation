// OutputPath.h
//
// Tiny cross-platform "mkdir -p"-equivalent. The original projects called
// POSIX mkdir()/sys/stat.h directly, which doesn't compile under MSVC.
// std::filesystem (C++17) covers Linux, macOS, and native Windows alike.

#pragma once

#include <filesystem>
#include <string>

namespace output_path {

inline void ensureDirectoryExists(const std::string& path) {
    std::filesystem::create_directories(path);
}

}  // namespace output_path
