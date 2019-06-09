function [  ] = edit_file_permission( file, value )

cmd = sprintf('chmod %s %s',value, file);
system(cmd);

end

