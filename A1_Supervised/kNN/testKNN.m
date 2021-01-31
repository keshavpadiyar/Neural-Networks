k = 2;

data = [4,5,0;3,2,1;6,3,1;7,8,0];

y = data(:,3);

%x = [3,3;4,5;5,2;6,9];

x = [4,5;3,2;6,3;7,8];


[d_nrow, d_ncol] = size(data);

[x_nrow, x_ncol] = size(x);

dist = [];
final_class = [];

for x_index = 1:x_nrow
 for d_index = 1:d_nrow
	dist(d_index) = sum(sqrt((data(d_index,1:2)-x(x_index,:)).^2)); % Eucledian Distance
 end
[d,i] = sort(dist);
i = i(1:k);
[c,v] = groupcounts(y(i));
v = v(find(c==max(c),1)); % Majority Voting
final_class(x_index) = v;
end