#include <iostream>
#include "author.h"

using namespace std;

namespace qgl {

	Author::Author() {
		/* default constructor */

		author_id = 0;
		strcpy(first_name, "null_first_name");
		strcpy(last_name, "null_last_name");
	}

	Author::~Author() {
		/* default desctructor */
	}

	void Author::print_details() {
		cout << "Author ID: " << author_id << endl;
		printf("First Name: %s\n", first_name);
		printf("Last Name: %s\n", first_name);
	}

	int Author::get_author_id() {
		return author_id;
	}

	string Author::get_first_name() {
		return first_name;
	}

	string Author::get_last_name() {
		return last_name;
	}
}