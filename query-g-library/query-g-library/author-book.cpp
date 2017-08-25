

#include "author-book.h"
#include <string.h>

using namespace std;

namespace qgl {
	AuthorBook::AuthorBook() {
		strcpy(first_name, "fn");
		strcpy(last_name, "ln");

		author_author_id = 0;
		book_author_id = 0;

		invertar_br = 0;
		strcpy(naslov, "ttl");
	}

	AuthorBook::~AuthorBook() {

	}

	void AuthorBook::print_details() {
		cout << "[author.xml] Author ID: " << author_author_id << endl;
		printf("[author.xml] First Name: %s\n", first_name);
		printf("[author.xml] Last Name: %s\n", last_name);

		cout << "[book.xml] Inventar br: " << invertar_br << endl;
		printf("[book.xml] Title: %s\n", naslov);
		cout << "[book.xml] Authoir ID: " << book_author_id << endl;
	}
}