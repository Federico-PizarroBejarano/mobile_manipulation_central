#include <iostream>
#include <vector>
#include <algorithm>  // For std::lower_bound
#include <numeric>    // For std::accumulate
template<typename T>
class FixedSizeVector {
private:
    std::vector<T> elements;  // Vector to hold the elements
    size_t max_size;          // Maximum size of the vector

public:
    // Constructor to initialize the fixed-size vector
    FixedSizeVector(size_t size) : max_size(size) {}

    // Add an element to the vector (assuming elements are added in the desired order)
    void add(T value) {
        if (elements.size() == max_size) {
            // If the vector is full, remove the oldest element (the one at index 0)
            elements.erase(elements.begin());
        }
        elements.push_back(value);  // Just add the element to the back
    }

    // Print all elements in the vector
    void print() const {
        for (const auto& val : elements) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Method to return a subvector and the index of the first element in the original vector
    std::pair<std::vector<T>, int> getSubVectorLargerThan(T num) const {
        // Find the largest element smaller than `num`
        auto it = std::lower_bound(elements.begin(), elements.end(), num);

        // If no element is smaller than `num`, return an empty vector and index -1
        if (it == elements.begin()) {
            return {std::vector<T>(), -1};  // Return empty vector and invalid index
        }

        // Calculate the index of the first element in the original vector
        int index = std::distance(elements.begin(), it);

        // Create a subvector starting from the first larger element
        return {std::vector<T>(it, elements.end()), index};
    }

    std::vector<T> getSubVectorFromIndex(int startIndex) {
        // Check if the startIndex is within valid bounds
        if (startIndex < 0 || startIndex >= elements.size()) {
            std::cerr << "Invalid start index!" << std::endl;
            return {};  // Return an empty vector for invalid index
        }

        // Use vector iterators to create a subvector starting from startIndex
        return std::vector<T>(elements.begin() + startIndex, elements.end());
    }

    // Method to get an element by index
    T getElementByIndex(size_t index) const {
        if (index < elements.size()) {
            return elements[index];  // Return the element at the specified index
        } else {
            throw std::out_of_range("Index out of range!");  // Handle out-of-bounds access
        }
    }

    // Method to calculate the average of the elements
    T avg() const {
        if (elements.empty()) {
            throw std::runtime_error("No elements in the vector to calculate the average");
        }
        // Calculate the sum using std::accumulate and divide by the number of elements
        T sum = std::accumulate(elements.begin(), elements.end(), T(0,0,0));
        
        return sum / elements.size();
    }
};