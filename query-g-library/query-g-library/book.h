#pragma once

#ifndef QUERY_G_BOOK_H
#define QUERY_G_BOOK_H
#include <iostream>

using namespace std;
namespace qgl {

	// All declarations are within the namespace scope.
	// Notice the lack of indentation.
	class Book {
	public:
			Book();
			~Book();
			void print_details();

			string get_title();
			int get_author_id();
			int get_invertar_id();

			int set_title(string new_title);
			int set_author_id(int new_id);
			int set_invertar_id(int new_id);

			string title;
			int author_id;
			int invertar_id;
	
	private:
//		string title;
//		int author_id;

	};

}  // namespace qgl

#endif  // QUERY_G_BOOK_H