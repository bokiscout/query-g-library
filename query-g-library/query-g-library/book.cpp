#include <iostream>
#include "book.h"

using namespace std;

namespace qgl {

	/* default constructor */
	Book::Book() {
		author_id = 0;
		title = "Empty title";
	}

	void Book::print_details() {
		//cout << "Title: " << title << endl << "Author ID: " << author_id << endl;
		cout << "ID: " << author_id << endl;
	}

	int Book::get_author_id() {
		return author_id;
	}

	int Book::set_author_id(int new_id) {
		author_id = new_id;
		
		return 1;
	}
}