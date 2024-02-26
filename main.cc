
#include <benchmark/benchmark.h>

#include <bitset>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "roaring/roaring.hh"
#include "sets.h"

constexpr uint32_t kNumSetElems = 65536;

// Create a set with members between
// [`start_member`, `start_member` + `num_members`).
// Each element has `probability` of being included.
absl::StatusOr<std::vector<uint32_t>> GetSetMembers(uint32_t start_member,
                                                    uint32_t num_members,
                                                    double probablity) {
    if (probablity > 1 || probablity < 0) {
        return absl::InvalidArgumentError("Probablity must be >= 0 and <= 1");
    }
    absl::BitGen gen;
    std::vector<uint32_t> result;
    for (auto i = start_member; i < start_member + num_members; i++) {
        if (absl::Bernoulli(gen, probablity)) {
            result.push_back(i);
        }
    }
    return result;
}

absl::StatusOr<std::pair<std::vector<uint32_t>, std::vector<uint32_t>>>
GetPairOfSetMembers(double probability) {
    auto left = GetSetMembers(0, kNumSetElems, 1);
    if (!left.ok()) {
        return left.status();
    }
    auto right = GetSetMembers(kNumSetElems / 2, kNumSetElems / 2, 1);
    if (!right.ok()) {
        return right.status();
    }
    return std::make_pair(std::move(*left), std::move(*right));
}

// The bytes of an integer representation of the value in a string.
// The length of the byte array is the sizeof each set member.
absl::StatusOr<
    std::pair<std::unordered_set<std::string>, std::unordered_set<std::string>>>
GetUnorderedStringSetsHelper(
    absl::AnyInvocable<std::string(uint32_t)> converter) {
    auto sets = GetPairOfSetMembers(1);
    if (!sets.ok()) {
        return sets.status();
    }
    auto [left, right] = std::move(*sets);
    std::unordered_set<std::string> left_set;
    for (auto v : left) {
        left_set.insert(converter(v));
    }
    std::unordered_set<std::string> right_set;
    for (auto v : right) {
        right_set.insert(converter(v));
    }
    return std::make_pair(std::move(left_set), std::move(right_set));
}

absl::StatusOr<std::pair<std::bitset<kNumSetElems>, std::bitset<kNumSetElems>>>
GetUnorderedBitSets() {
    auto sets = GetPairOfSetMembers(1);
    if (!sets.ok()) {
        return sets.status();
    }
    auto [left, right] = std::move(*sets);
    std::bitset<kNumSetElems> left_set;
    for (auto v : left) {
        left_set.set(v);
    }
    std::bitset<kNumSetElems> right_set;
    for (auto v : right) {
        right_set.set(v);
    }
    return std::make_pair(std::move(left_set), std::move(right_set));
}

absl::StatusOr<std::pair<roaring::Roaring, roaring::Roaring>> GetRoaringSets() {
    auto sets = GetPairOfSetMembers(1);
    if (!sets.ok()) {
        return sets.status();
    }
    auto [left, right] = std::move(*sets);
    roaring::Roaring left_set;
    for (auto v : left) {
        left_set.add(v);
    }
    roaring::Roaring right_set;
    for (auto v : right) {
        right_set.add(v);
    }
    left_set.runOptimize();
    right_set.runOptimize();
    return std::make_pair(std::move(left_set), std::move(right_set));
}

template <typename T, typename U>
void BenchmarkSetHelper(benchmark::State& state, T set_getter, U op) {
    auto sets = set_getter();
    ASSERT_TRUE(sets.ok());
    for (auto _ : state) {
        state.PauseTiming();
        auto [left, right] = *sets;
        state.ResumeTiming();
        auto result = op(std::move(left), std::move(right));
        benchmark::DoNotOptimize(result);
    }
}

class UnorderedIntSet {
   public:
    absl::StatusOr<
        std::pair<std::unordered_set<uint32_t>, std::unordered_set<uint32_t>>>
    operator()() {
        auto sets = GetPairOfSetMembers(1);
        if (!sets.ok()) {
            return sets.status();
        }
        auto [left, right] = std::move(*sets);
        std::unordered_set<uint32_t> left_set(left.begin(), left.end());
        std::unordered_set<uint32_t> right_set(right.begin(), right.end());
        return std::make_pair(std::move(left_set), std::move(right_set));
    }
};

// The bytes of an integer representation of the value in a string.
// The length of the byte array is the sizeof each set member.
class UnorderedByteSet {
   public:
    absl::StatusOr<std::pair<std::unordered_set<std::string>,
                             std::unordered_set<std::string>>>
    operator()() {
        return GetUnorderedStringSetsHelper([](uint32_t val) {
            return std::string(reinterpret_cast<const char*>(&val),
                               sizeof(val));
        });
    }
};

// The integer represented as a string.
// Demonstrates the implications of a longer string value.
class UnorderedStringSet {
   public:
    absl::StatusOr<std::pair<std::unordered_set<std::string>,
                             std::unordered_set<std::string>>>
    operator()() {
        return GetUnorderedStringSetsHelper(
            [](uint32_t val) { return std::to_string(val); });
    }
};

class BitSet {
   public:
    absl::StatusOr<
        std::pair<std::bitset<kNumSetElems>, std::bitset<kNumSetElems>>>
    operator()() {
        auto sets = GetPairOfSetMembers(1);
        if (!sets.ok()) {
            return sets.status();
        }
        auto [left, right] = std::move(*sets);
        std::bitset<kNumSetElems> left_set;
        for (auto v : left) {
            left_set.set(v);
        }
        std::bitset<kNumSetElems> right_set;
        for (auto v : right) {
            right_set.set(v);
        }
        return std::make_pair(std::move(left_set), std::move(right_set));
    }
};

class RoaringBitSet {
   public:
    absl::StatusOr<std::pair<roaring::Roaring, roaring::Roaring>> operator()() {
        auto sets = GetPairOfSetMembers(1);
        if (!sets.ok()) {
            return sets.status();
        }
        auto [left, right] = std::move(*sets);
        roaring::Roaring left_set;
        for (auto v : left) {
            left_set.add(v);
        }
        roaring::Roaring right_set;
        for (auto v : right) {
            right_set.add(v);
        }
        left_set.runOptimize();
        right_set.runOptimize();
        return std::make_pair(std::move(left_set), std::move(right_set));
    }
};

template <typename T>
class Op {
   public:
    typedef std::tuple_element<
        0, typename std::invoke_result_t<T>::value_type>::type value_type;
};

template <typename T>
class UnionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return Union(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseUnionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return left | right;
    }
};

template <typename T>
class IntersectionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return Intersection(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseIntersectionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return left & right;
    }
};

template <typename T>
class DifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return Difference(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseDifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return (left & (~right));
    }
};

template <typename T>
class RoaringBitwiseDifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) {
        return (left - right);
    }
};

template <typename T, typename U>
class SetOpFixture : public benchmark::Fixture {
   protected:
    T getter_;
    U op_;
};

// Unordered set of integers
BENCHMARK_TEMPLATE_F(SetOpFixture, IntTestUnion, UnorderedIntSet,
                     UnionOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, IntTestIntersection, UnorderedIntSet,
                     IntersectionOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, IntTestDifference, UnorderedIntSet,
                     DifferenceOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

// Unordered set of bytes
BENCHMARK_TEMPLATE_F(SetOpFixture, ByteTestUnion, UnorderedByteSet,
                     UnionOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, ByteTestIntersection, UnorderedByteSet,
                     IntersectionOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, ByteTestDifference, UnorderedByteSet,
                     DifferenceOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

// Unordered set of strings
BENCHMARK_TEMPLATE_F(SetOpFixture, StringTestUnion, UnorderedStringSet,
                     UnionOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, StringTestIntersection, UnorderedStringSet,
                     IntersectionOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, StringTestDifference, UnorderedStringSet,
                     DifferenceOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

// std::bitset implementation
BENCHMARK_TEMPLATE_F(SetOpFixture, StdBitSetTestUnion, BitSet,
                     BitwiseUnionOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, StdBitSetTestIntersection, BitSet,
                     BitwiseIntersectionOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, StdBitSetTestDifference, BitSet,
                     BitwiseDifferenceOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

// Roaring bitset implementation
BENCHMARK_TEMPLATE_F(SetOpFixture, RoaringBitSetTestUnion, RoaringBitSet,
                     BitwiseUnionOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, RoaringBitSetTestIntersection, RoaringBitSet,
                     BitwiseIntersectionOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

BENCHMARK_TEMPLATE_F(SetOpFixture, RoaringBitSetTestDifference, RoaringBitSet,
                     RoaringBitwiseDifferenceOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }

// Run the benchmark
BENCHMARK_MAIN();