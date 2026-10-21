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

    // Volta (sm_70): the table is there, and every switch in it is one of the measured ones
    {
        const auto t = nv_default_env(7, 0);
        CHECK(!t.empty());
        bool sm70_table = false;
        for (const auto& kv : t) {
            CHECK(kv.first.rfind("STRATA_", 0) == 0);
            CHECK(kv.second == "1" || kv.second == "2");
            if (kv.first == "STRATA_SM70_TABLE") { sm70_table = true; CHECK(kv.second == "1"); }
            // a switch that changes the answer has no place in a table of exact switches
            CHECK(kv.first != "STRATA_PF_FUSED" && kv.first != "STRATA_PF_GEMM" && kv.first != "STRATA_HCD_EXACT" &&
                  kv.first != "STRATA_EXCHANGE_ROTATE" && kv.first != "STRATA_ROUTE_RESIDENT" &&
                  kv.first != "STRATA_ROUTE_TAIL_SKIP" && kv.first != "STRATA_PCIE_BALANCE");
        }
        CHECK(sm70_table);   // the Volta master switch is in the table
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

    // applying twice sets nothing new the second time
    {
        const auto first = apply_nv_defaults(7, 0);
        CHECK(!first.empty());
        const auto second = apply_nv_defaults(7, 0);
        CHECK(second.empty());
        for (const auto& s : first) unset(s.c_str());
    }

    if (fails == 0) std::printf("nv_defaults_test: PASS\n");
    else std::printf("nv_defaults_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
