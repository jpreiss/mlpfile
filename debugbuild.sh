clang++ \
-c \
--std=c++17 \
-I`python3 -c "import eigenpip; print(eigenpip.get_include())"` \
mlpfile/cpp/mlpfile.cpp
