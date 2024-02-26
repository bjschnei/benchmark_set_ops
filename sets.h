#include <utility>

template <typename T>
T Union(T&& left, T&& right) {
    auto& small = left.size() <= right.size() ? left : right;
    auto& big = left.size() <= right.size() ? right : left;
    big.insert(small.begin(), small.end());
    return std::move(big);
}

template <typename T>
T Intersection(T&& left, T&& right) {
    auto& small = left.size() <= right.size() ? left : right;
    const auto& big = left.size() <= right.size() ? right : left;
    // Traverse the smaller set removing what is not in both.
    //absl::erase_if(small,
                   //[&big](const auto& elem) { return !big.contains(elem); });
    std::erase_if(small,
                  [&big](const auto& elem) { return !big.contains(elem); });
    return std::move(small);
}

template <typename T>
T Difference(T&& left, T&& right) {
    // Remove all elements in right from left.
    for (const auto& element : right) {
        left.erase(element);
    }
    return std::move(left);
}
