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
#include "bm.h"
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
    auto left = GetSetMembers(0, kNumSetElems, probability);
    if (!left.ok()) {
        return left.status();
    }
    auto right = GetSetMembers(kNumSetElems / 2, kNumSetElems / 2, probability);
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
    absl::AnyInvocable<std::string(uint32_t)> converter, double probability) {
    auto sets = GetPairOfSetMembers(probability);
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

// Template used to run benchmarks for all set types and operations.
template <typename T, typename U>
void BenchmarkSetHelper(benchmark::State& state, const T& set_getter,
                        const U& op) {
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

class BaseGetter {
   public:
    void SetProbabilityPercent(uint8_t probability_percent) {
        probability_ = probability_percent / 100.0;
    }

    double GetProbability() const { return probability_; }

   private:
    double probability_ = 1;
};

class UnorderedIntSet : public BaseGetter {
   public:
    absl::StatusOr<
        std::pair<std::unordered_set<uint32_t>, std::unordered_set<uint32_t>>>
    operator()() const {
        auto sets = GetPairOfSetMembers(GetProbability());
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
class UnorderedByteSet : public BaseGetter {
   public:
    absl::StatusOr<std::pair<std::unordered_set<std::string>,
                             std::unordered_set<std::string>>>
    operator()() const {
        return GetUnorderedStringSetsHelper(
            [](uint32_t val) {
                return std::string(reinterpret_cast<const char*>(&val),
                                   sizeof(val));
            },
            GetProbability());
    }
};

// The integer represented as a string.
// Demonstrates the implications of a longer string value.
class UnorderedStringSet : public BaseGetter {
   public:
    absl::StatusOr<std::pair<std::unordered_set<std::string>,
                             std::unordered_set<std::string>>>
    operator()() const {
        return GetUnorderedStringSetsHelper(
            [](uint32_t val) { return std::to_string(val); }, GetProbability());
    }
};

class BitSet : public BaseGetter {
   public:
    absl::StatusOr<
        std::pair<std::bitset<kNumSetElems>, std::bitset<kNumSetElems>>>
    operator()() const {
        auto sets = GetPairOfSetMembers(GetProbability());
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

class RoaringBitSet : public BaseGetter {
   public:
    absl::StatusOr<std::pair<roaring::Roaring, roaring::Roaring>> operator()()
        const {
        auto sets = GetPairOfSetMembers(GetProbability());
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

class BitmagicBitSet : public BaseGetter {
   public:
    absl::StatusOr<std::pair<bm::bvector<>, bm::bvector<>>> operator()() const {
        auto sets = GetPairOfSetMembers(GetProbability());
        if (!sets.ok()) {
            return sets.status();
        }
        auto [left, right] = std::move(*sets);
        bm::bvector<> left_set;
        left_set.init();
        for (auto v : left) {
            left_set.set_bit(v);
        }
        bm::bvector<> right_set;
        right_set.init();
        for (auto v : right) {
            right_set.set_bit(v);
        }
        left_set.optimize();
        right_set.optimize();
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
                                 Op<T>::value_type&& right) const {
        return Union(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseUnionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return left | right;
    }
};

template <typename T>
class IntersectionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return Intersection(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseIntersectionOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return left & right;
    }
};

template <typename T>
class DifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return Difference(std::move(left), std::move(right));
    }
};

template <typename T>
class BitwiseDifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return (left & (~right));
    }
};

template <typename T>
class RoaringBitwiseDifferenceOp {
   public:
    Op<T>::value_type operator()(Op<T>::value_type&& left,
                                 Op<T>::value_type&& right) const {
        return (left - right);
    }
};

// Fixture that sets the probablity based on the test parameter.
template <typename T, typename U>
class SetOpFixture : public benchmark::Fixture {
   public:
    void SetUp(::benchmark::State& state) {
        getter_.SetProbabilityPercent(state.range(0));
    }

   protected:
    T getter_;
    U op_;
};

// Fixture for iterating across a set and returning all set elements
template <typename T>
class GetSetElementsFixture : public benchmark::Fixture {
   public:
    void SetUp(::benchmark::State& state) {
        getter_.SetProbabilityPercent(state.range(0));
    }

   protected:
    T getter_;
};

void GetSetElements(benchmark::State& state,
                    const UnorderedIntSet& set_getter) {
    auto sets = set_getter();
    ASSERT_TRUE(sets.ok());
    const auto& set = sets->first;
    for (auto _ : state) {
        std::vector<uint32_t> result;
        for (const auto& val : set) {
            result.push_back(val);
        }
        benchmark::DoNotOptimize(result);
    }
}

void GetSetElements(benchmark::State& state, const RoaringBitSet& set_getter) {
    auto sets = set_getter();
    ASSERT_TRUE(sets.ok());
    const auto& set = sets->first;
    for (auto _ : state) {
        std::vector<uint32_t> result;
        result.reserve(set.cardinality());
        set.toUint32Array(result.data());
        benchmark::DoNotOptimize(result);
    }
}

void GetSetElements(benchmark::State& state, const BitmagicBitSet& set_getter) {
    auto sets = set_getter();
    ASSERT_TRUE(sets.ok());
    const auto& set = sets->first;
    for (auto _ : state) {
        std::vector<uint32_t> result;
        auto val = set.get_first();
        if (val == 0) {
            if (!set.get_bit(0)) {
                continue;
            }
        }
        result.push_back(val);
        while (true) {
            val = set.get_next(val);
            if (!val) {
                break;
            }
            result.push_back(val);
        }
        benchmark::DoNotOptimize(result);
    }
}

void GetSetElements(benchmark::State& state, const BitSet& set_getter) {
    auto sets = set_getter();
    ASSERT_TRUE(sets.ok());
    const auto& set = sets->first;
    for (auto _ : state) {
        std::vector<uint32_t> result;
        for (auto i = 0; i < kNumSetElems; i++) {
            if (set.test(i)) {
                result.push_back(i);
            }
        }
        benchmark::DoNotOptimize(result);
    }
}

// Test how quickly we can do set operations.
// Unordered set of integers
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, IntTestUnion, UnorderedIntSet,
                            UnionOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, IntTestUnion)->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, IntTestIntersection, UnorderedIntSet,
                            IntersectionOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, IntTestIntersection)->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, IntTestDifference, UnorderedIntSet,
                            DifferenceOp<UnorderedIntSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, IntTestDifference)->DenseRange(0, 100, 25);

// Unordered set of bytes
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, ByteTestUnion, UnorderedByteSet,
                            UnionOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, ByteTestUnion)->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, ByteTestIntersection,
                            UnorderedByteSet, IntersectionOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, ByteTestIntersection)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, ByteTestDifference, UnorderedByteSet,
                            DifferenceOp<UnorderedByteSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, ByteTestDifference)->DenseRange(0, 100, 25);

// Unordered set of strings
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StringTestUnion, UnorderedStringSet,
                            UnionOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StringTestUnion)->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StringTestIntersection,
                            UnorderedStringSet,
                            IntersectionOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StringTestIntersection)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StringTestDifference,
                            UnorderedStringSet,
                            DifferenceOp<UnorderedStringSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StringTestDifference)
    ->DenseRange(0, 100, 25);

// std::bitset implementation
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StdBitSetTestUnion, BitSet,
                            BitwiseUnionOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StdBitSetTestUnion)->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StdBitSetTestIntersection, BitSet,
                            BitwiseIntersectionOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StdBitSetTestIntersection)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, StdBitSetTestDifference, BitSet,
                            BitwiseDifferenceOp<BitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, StdBitSetTestDifference)
    ->DenseRange(0, 100, 25);

// Roaring bitset implementation
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, RoaringBitSetTestUnion, RoaringBitSet,
                            BitwiseUnionOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, RoaringBitSetTestUnion)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, RoaringBitSetTestIntersection,
                            RoaringBitSet, BitwiseIntersectionOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, RoaringBitSetTestIntersection)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, RoaringBitSetTestDifference,
                            RoaringBitSet,
                            RoaringBitwiseDifferenceOp<RoaringBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, RoaringBitSetTestDifference)
    ->DenseRange(0, 100, 25);

// bitmagic bitset implementation
BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, BitmagicBitSetTestUnion,
                            BitmagicBitSet, BitwiseUnionOp<BitmagicBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, BitmagicBitSetTestUnion)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, BitmagicBitSetTestIntersection,
                            BitmagicBitSet,
                            BitwiseIntersectionOp<BitmagicBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, BitmagicBitSetTestIntersection)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(SetOpFixture, BitmagicBitSetTestDifference,
                            BitmagicBitSet,
                            RoaringBitwiseDifferenceOp<BitmagicBitSet>)
(benchmark::State& state) { BenchmarkSetHelper(state, getter_, op_); }
BENCHMARK_REGISTER_F(SetOpFixture, BitmagicBitSetTestDifference)
    ->DenseRange(0, 100, 25);

// Test how quickly we can read the set bits into a vector of uint32
BENCHMARK_TEMPLATE_DEFINE_F(GetSetElementsFixture, IntTestElems,
                            UnorderedIntSet)
(benchmark::State& state) { GetSetElements(state, getter_); }
BENCHMARK_REGISTER_F(GetSetElementsFixture, IntTestElems)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(GetSetElementsFixture, StdBitSetTestElems, BitSet)
(benchmark::State& state) { GetSetElements(state, getter_); }
BENCHMARK_REGISTER_F(GetSetElementsFixture, StdBitSetTestElems)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(GetSetElementsFixture, RoaringTestElems,
                            RoaringBitSet)
(benchmark::State& state) { GetSetElements(state, getter_); }
BENCHMARK_REGISTER_F(GetSetElementsFixture, RoaringTestElems)
    ->DenseRange(0, 100, 25);

BENCHMARK_TEMPLATE_DEFINE_F(GetSetElementsFixture, BitmagicTestElems,
                            BitmagicBitSet)
(benchmark::State& state) { GetSetElements(state, getter_); }
BENCHMARK_REGISTER_F(GetSetElementsFixture, BitmagicTestElems)
    ->DenseRange(0, 100, 25);

// Run the benchmark
BENCHMARK_MAIN();
