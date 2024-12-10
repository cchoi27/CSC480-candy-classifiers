#ifndef CANDY_DATA_H
#define CANDY_DATA_H

// Define a struct to store candy information
struct Candy {
    const char* name; // Candy name
    int lactose_flag;        // First flag
    int nuts_flag;        // Second flag
};

const Candy candyArray[] = {
    {"Twix", 1, 0},
    {"Skittles", 0, 0},
    {"Snickers", 1, 1},
    {"Nothing", 0, 0},
    {"nothing??", 0, 0}

};

const int candyCount = sizeof(candyArray) / sizeof(Candy);

bool getCandyFlags(const char* candyName, int &flag1, int &flag2) {
    for (int i = 0; i < candyCount; i++) {
        if (strcmp(candyArray[i].name, candyName) == 0) {
            flag1 = candyArray[i].lactose_flag;
            flag2 = candyArray[i].nuts_flag;
            return true; // Candy found
        }
    }
    return false;
}

#endif // CANDY_DATA_H
