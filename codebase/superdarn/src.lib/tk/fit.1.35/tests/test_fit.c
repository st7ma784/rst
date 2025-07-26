#include <gtest/gtest.h>
#include "fitdata.h"
#include "fitblk.h"

// Test fixture for FIT module tests
class FitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data here if needed
    }

    void TearDown() override {
        // Clean up test data here if needed
    }
};

// Basic test to verify the test framework is working
TEST_F(FitTest, BasicTest) {
    EXPECT_EQ(1, 1);
}

// Test CFitMake and CFitFree
TEST_F(FitTest, CreateAndFree) {
    struct CFitdata *cfit = CFitMake();
    EXPECT_NE(cfit, nullptr);
    CFitFree(cfit);
}

// Test CFitSetRng
TEST_F(FitTest, SetRanges) {
    struct CFitdata *cfit = CFitMake();
    const int test_ranges = 10;
    
    CFitSetRng(cfit, test_ranges);
    EXPECT_EQ(cfit->nrang, test_ranges);
    EXPECT_NE(cfit->rng, nullptr);
    
    // Test setting zero ranges
    CFitSetRng(cfit, 0);
    EXPECT_EQ(cfit->nrang, 0);
    EXPECT_EQ(cfit->rng, nullptr);
    
    CFitFree(cfit);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
