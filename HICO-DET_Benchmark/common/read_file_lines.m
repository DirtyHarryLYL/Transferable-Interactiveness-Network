function [ lines ] = read_file_lines( file, flag_indent )

if nargin < 2
    flag_indent = false;
end

fid = fopen(file,'r');
if flag_indent
    lines = textscan(fid,'%s',-1,'delimiter',{'\n'},'whitespace','');
else
    lines = textscan(fid,'%s',-1,'delimiter',{'\n'});
end
lines = lines{1};
fclose(fid);

end

