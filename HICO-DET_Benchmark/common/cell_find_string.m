function [ id ] = cell_find_string( C, string )

tind = cellfun(@(x)strcmp(x,string),C);
id = find(tind);

end

