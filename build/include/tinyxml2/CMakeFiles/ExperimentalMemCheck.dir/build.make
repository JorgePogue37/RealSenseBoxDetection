# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jorge/Documents/RealsenseBoxDetection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jorge/Documents/RealsenseBoxDetection/build

# Utility rule file for ExperimentalMemCheck.

# Include the progress variables for this target.
include include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/progress.make

include/tinyxml2/CMakeFiles/ExperimentalMemCheck:
	cd /home/jorge/Documents/RealsenseBoxDetection/build/include/tinyxml2 && /usr/bin/ctest -D ExperimentalMemCheck

ExperimentalMemCheck: include/tinyxml2/CMakeFiles/ExperimentalMemCheck
ExperimentalMemCheck: include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/build.make

.PHONY : ExperimentalMemCheck

# Rule to build all files generated by this target.
include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/build: ExperimentalMemCheck

.PHONY : include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/build

include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/clean:
	cd /home/jorge/Documents/RealsenseBoxDetection/build/include/tinyxml2 && $(CMAKE_COMMAND) -P CMakeFiles/ExperimentalMemCheck.dir/cmake_clean.cmake
.PHONY : include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/clean

include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/depend:
	cd /home/jorge/Documents/RealsenseBoxDetection/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jorge/Documents/RealsenseBoxDetection /home/jorge/Documents/RealsenseBoxDetection/include/tinyxml2 /home/jorge/Documents/RealsenseBoxDetection/build /home/jorge/Documents/RealsenseBoxDetection/build/include/tinyxml2 /home/jorge/Documents/RealsenseBoxDetection/build/include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : include/tinyxml2/CMakeFiles/ExperimentalMemCheck.dir/depend

