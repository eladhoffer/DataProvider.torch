package = "DataProvider"
version = "scm-1"

source = {
   url = "https://github.com/eladhoffer/DataProvider.torch.git"
   tag = "master"
}

description = {
   summary = "Data Provider Tools Made By Elad Hoffer",
   detailed = [[
   	    Data Provider for torch
   ]],
   homepage = "https://github.com/ehoffer/DataProvider.torch"
}

dependencies = {

}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)";
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
