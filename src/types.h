#pragma once

#include <cassert>
#include <algorithm> // for std::copy
#include <limits>

class Array 
{
public:
    Array() : size(0), data(nullptr) {}

    Array(int size) : size(size) {
        data = new float[size]{};
    }

    // for initializer list
    Array(std::initializer_list<float> initList) 
    : size(static_cast<int>(initList.size())) {
        data = new float[size];
        std::copy(initList.begin(), initList.end(), data);
    }

    // copy constructor
    Array(const Array& other) : size(other.size) {
        data = new float[size];
        std::copy(other.data, other.data + size, data);
    }

    // assignment operator
    Array& operator=(const Array& other) {
        if (this != &other) {
            delete[] data;
            size = other.size;
            data = new float[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

    ~Array() {
        delete[] data;
    }

    int getSize() const {
        return size;
    }

    float& operator[](int index) {
        assert(data != nullptr);
        assert(index >= 0 && index < size);
        return data[index];
    }

    const float& operator[](int index) const {
        assert(data != nullptr);
        assert(index >= 0 && index < size);
        return data[index];
    }

private:
    int size;
    float* data;
};
