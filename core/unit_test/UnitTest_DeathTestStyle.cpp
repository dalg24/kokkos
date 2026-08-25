// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>

#include <iostream>

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  ::testing::InitGoogleTest(&argc, argv);
  std::cout << GTEST_FLAG_GET(death_test_style) << "\n";
  return 0;
}
