#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_set>
using namespace std;

int T;
int a[10][9][9];
int main() {
    cin >> T;
   for (int i = 0; i < T; ++i) {
       bool checked = false;
       for (int j = 0; j < 9; ++j) {
           for (int k = 0; k < 9; ++k) {
               cin >> a[i][j][k];
           }
       }
       for (int j = 0; j < 9; ++j) {
           for (int k = 0; k < 9; ++k) {
              if (a[i][j][k] < 1 || a[i][j][k] > 9) {
                  cout << "Invalid" << endl;
                  checked = true;
                  break;
              }
           }
           if (checked) {
               break;
           }
       }

        if (checked) {
            continue;
        }
       for (int j = 0; j < 9; ++j) {
           unordered_set<int> tmp1;
           for (int k = 0; k < 9; ++k) {
               if (tmp1.find(a[i][j][k]) == tmp1.end()) {
                   tmp1.insert(a[i][j][k]);
               } else {
                   cout << "Invalid" << endl;
                   checked = true;
                   break;
               }
           }
           if (checked) {
               break;
           }
       }
        if (checked) {
           continue;
       }
       for (int j = 0; j < 9; ++j) {
           unordered_set<int> tmp1;
           for (int k = 0; k < 9; ++k) {
               if (tmp1.find(a[i][k][j]) == tmp1.end()) {
                   tmp1.insert(a[i][k][j]);
               } else {
                   cout << "Invalid" << endl;
                   checked = true;
                   break;
               }
           }
           if (checked) {
               break;
           }
       }

       if (checked) {
           continue;
       }
       
       for (int j = 0; j < 9; j+=3) {
           for (int k = 0; k < 9; k+=3) {
               unordered_set<int> tmp3;
               for (int c = j; c < 3 + j; ++c) {
                   for (int b = k; b < 3 + k; ++b) {
                       if (tmp3.find(a[i][c][b]) == tmp3.end()) {
                           tmp3.insert(a[i][c][b]);
                       } else {
                           cout << i << " " << c << " " << b << endl;
                           cout << "Invalid" << endl;
                           checked = true;
                           break;
                       }
                   }
                   if (checked) {
                       break;
                   }
               }
                if (checked) {
                     break;
                }
           }
            if (checked) {
            break;
            }
       }
       if (!checked) {
           cout << "Valid" << endl;
       }
   }
}
