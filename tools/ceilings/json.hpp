#pragma once

// Minimal JSON writer shared by the CPU and PCIe halves of the probe. The
// report is one object built once at exit, so plain string assembly suffices.
// snprintf rather than std::format keeps this header digestible by nvcc.

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pg_ceil {

inline std::string json_str(std::string_view s) {
    std::string out = "\"";
    for (const char ch : s) {
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (static_cast<unsigned char>(ch) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned>(ch));
            out += buf;
        } else {
            out += ch;
        }
    }
    return out + "\"";
}

// NaN and Inf are not JSON; a failed or degenerate measurement reads as null.
inline std::string num(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", v);
    return buf;
}

inline std::string join(const std::vector<std::string>& items) {
    std::string out = "[";
    for (const std::string& item : items) {
        if (out.size() > 1) {
            out += ',';
        }
        out += item;
    }
    return out + "]";
}

// Values arrive already encoded, so callers choose json_str, num, or a literal.
inline std::string obj(std::initializer_list<std::pair<std::string_view, std::string>> kv) {
    std::string out = "{";
    for (const auto& [key, value] : kv) {
        if (out.size() > 1) {
            out += ',';
        }
        out += json_str(key);
        out += ':';
        out += value;
    }
    return out + "}";
}

} // namespace pg_ceil
