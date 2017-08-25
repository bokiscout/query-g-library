#pragma once

#ifndef QUERY_G_AUTHOR_H
#define QUERY_G_AUTHOR_H
#include <iostream>

using namespace std;
namespace qgl {

	// All declarations are within the namespace scope.
	// Notice the lack of indentation.
	class Author {
	public:
		Author();
		~Author();

		int author_id;
		char first_name[20];
		char last_name[20];

		void print_details();

		string get_first_name();
		string get_last_name();
		int get_author_id();

		int set_first_name(string new_name);
		int set_last_name(string new_surname);

	private:
		//		string title;
		//		int author_id;

	};

}  // namespace qgl

#endif  // QUERY_G_AUTHOR_H