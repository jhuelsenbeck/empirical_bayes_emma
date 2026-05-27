#include <cstdlib>
#include <iostream>
#include "Msg.hpp"



void Msg::error(std::string s) {

	std::cout << "Error: " << s << std::endl;
	std::cout << "Exiting program" << std::endl;
	std::exit(1);
}

void Msg::warning(std::string s) {

	std::cout << "Warning: " << s << std::endl;
}
