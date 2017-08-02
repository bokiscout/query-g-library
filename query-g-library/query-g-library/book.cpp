#include <iostream>
#include "book.h"

using namespace std;

namespace qgl {

	Book::Book() {
		/* default constructor */

		author_id = 0;
		invertar_id = 0;
		title = "Empty title";
	}

	Book::~Book() {
		/* default desctructor */
	}

	void Book::print_details() {
		//cout << "Title: " << title << endl << "Author ID: " << author_id << endl;
		cout << "Invertar ID: " << invertar_id << endl;
		cout << "Author ID: " << author_id << endl;
		//cout << "Title: " << title << endl;
		printf("Title: %s\n", title.c_str());
	}

	int Book::get_author_id() {
		return author_id;
	}

	string Book::get_title() {
		return title;
	}

	int Book::get_invertar_id() {
		return invertar_id;
	}

	int Book::set_author_id(int new_id) {
		author_id = new_id;
		
		return 1;
	}

	int Book::set_title(string new_title) {
		title = new_title;

		return 1;
	}

	int Book::set_invertar_id(int new_id) {
		invertar_id = new_id;

		return 1;
	}
}