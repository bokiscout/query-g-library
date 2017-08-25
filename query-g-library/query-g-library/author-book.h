#pragma once

#ifndef QUERY_G_AUTHOR_BOOK_H
#define QUERY_G_AUTHOR_BOOK_H
#include <iostream>

using namespace std;
namespace qgl {

	// All declarations are within the namespace scope.
	// Notice the lack of indentation.
	class AuthorBook {
	public:
		AuthorBook();
		~AuthorBook();

		int author_author_id;		// author ID of authors.xml
		char first_name[20];
		char last_name[20];

		int invertar_br;
		char naslov[50];
		int book_author_id;			// author ID of books.xml

		void print_details();

	private:
		//		string title;
		//		int author_id;

	};

}  // namespace qgl

#endif  // QUERY_G_AUTHOR_BOOK_H