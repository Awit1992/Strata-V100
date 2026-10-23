// src/core/nv_defaults_test.cpp - the per-architecture default table (no GPU needed).
#include "strata/core/nv_defaults.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static void unset(const char* k) {
#if defined(_WIN32)
    _putenv_s(k, "");
#else
    unsetenv(k);
#endif
}
static void put(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}

int main() {
    using namespace strata::core;
    unset("STRATA_SM70_PERF");
    unset("STRATA_SM86_PERF");
    unset("STRATA_SM60_PERF");

    // a card we do not have a table for: nothing
    CHECK(nv_default_env(8, 9).empty());
    CHECK(nv_default_env(7, 5).empty());
    CHECK(nv_default_env(6, 1).empty());

    // The new Volta policies must not change another architecture's process settings.
    const int other_arches[][2] = {{8, 6}, {6, 0}};
    for (const auto& cc : other_arches) {
        unset("STRATA_ADAPT_OVERLAP");
        unset("STRATA_PF_WMMA2");
        const auto applied = apply_nv_defaults(cc[0], cc[1]);
        CHECK(std::getenv("STRATA_ADAPT_OVERLAP") == nullptr);
        CHECK(std::getenv("STRATA_PF_WMMA2") == nullptr);
        for (const auto& key : applied) unset(key.c_str());
    }

    // the off switch empties the table, and the other architectures keep theirs
    put("STRATA_SM70_PERF", "0");
    CHECK(nv_default_env(7, 0).empty());
    CHECK(!nv_default_env(8, 6).empty());
    CHECK(!nv_default_env(6, 0).empty());
    unset("STRATA_SM70_PERF");

    // a switch the user set is kept, whatever its value
    put("STRATA_SM70_TABLE", "0");
    {
        const auto set = apply_nv_defaults(7, 0);
        for (const auto& s : set) CHECK(s != "STRATA_SM70_TABLE");
        const char* v = std::getenv("STRATA_SM70_TABLE");
        CHECK(v != nullptr && std::string(v) == "0");
    }
    unset("STRATA_SM70_TABLE");

    // an explicit STRATA_ADAPT_OVERLAP=0 is the user's own setting: kept, so the overlap stays off
    put("STRATA_ADAPT_OVERLAP", "0");
    {
        const auto set = apply_nv_defaults(7, 0);
        for (const auto& s : set) CHECK(s != "STRATA_ADAPT_OVERLAP");
        const char* v = std::getenv("STRATA_ADAPT_OVERLAP");
        CHECK(v != nullptr && std::string(v) == "0");
    }
    unset("STRATA_ADAPT_OVERLAP");

    // an explicit STRATA_PF_WMMA2=0 keeps the first tile only
    put("STRATA_PF_WMMA2", "0");
    {
        const auto set = apply_nv_defaults(7, 0);
        for (const auto& s : set) CHECK(s != "STRATA_PF_WMMA2");
        const char* v = std::getenv("STRATA_PF_WMMA2");
        CHECK(v != nullptr && std::string(v) == "0");
    }
    unset("STRATA_PF_WMMA2");

    // applying twice sets nothing new the second time
    {
        const auto first = apply_nv_defaults(7, 0);
        const auto second = apply_nv_defaults(7, 0);
        CHECK(second.empty());
        for (const auto& s : first) unset(s.c_str());
    }

    if (fails == 0) std::printf("nv_defaults_test: PASS\n");
    else std::printf("nv_defaults_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
