function [  ] = write_file_lines( file, lines )

fid = fopen(file,'w');
for i = 1:numel(lines)
    fprintf(fid,'%s',lines{i});
    if i ~= numel(lines)
        fprintf(fid,'\n');
    end
end
fclose(fid);

end

