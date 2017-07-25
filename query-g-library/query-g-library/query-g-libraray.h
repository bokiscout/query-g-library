#pragma once

#ifndef QUERY_G_LIBRARAY
#define QUERY_G_LIBRARAY
#include <iostream>
#include "book.h"

using namespace std;
namespace qgl {

	// All declarations are within the namespace scope.
	// Notice the lack of indentation.
	class QGLibraray {
	public:
		QGLibraray();
		~QGLibraray();

		Book *find_book_by_id(int id);
		void increse_book_id(Book *book, int amount);

	private:

	};

}  // namespace qgl

#endif  // QUERY_G_LIBRARAY