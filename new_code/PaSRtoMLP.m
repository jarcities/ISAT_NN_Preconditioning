DD = dlmread('output.dat'); % read the raw PaSR output

nSpecies = 163; nx = nSpecies+1; nLines = ceil(nx/3);

S = size(DD);

Nsamples = S(1)/(2*nLines); % calculate the number of samples

data = zeros(Nsamples,2*nx); 

for ii = 1:Nsamples

    dd = reshape(DD([1:nLines]+(ii-1)*2*nLines, :)',1,3*nLines);

    data(ii,1:nx) = dd(1:nx); %populate the 11 x-entries

    dd = reshape(DD([1:nLines]+nLines+(ii-1)*2*nLines, :)',1,3*nLines);

    data(ii,nx+1:2*nx) = dd(1:nx); % populate the 11 f(x) entries

end

I = randperm(Nsamples);

data = data(I,:); % randomly permute the samples

D = data;

N = 40000; % MLP training will be done on 40000 samples

ndim = nx; % dimension of x

lmin = 0.05; % minimum x-distance between two points in the MLP dataset

near = zeros(N,10); % work array

iPick = [1]; % always pick the first entry in the permuted data

count = 1; % number of samples picked so far

ii = 1; % number of samples considered so far

while count < N

    [count, ii]

    ii = ii+1; % index of the next point to be considered

    dist = (sum( (D(iPick,[1:ndim]) - D(ii,[1:ndim])).^2, 2)).^0.5; % calculate the distance between the 
    % current point and all other points which have been chosen

    if ( min( dist ) > lmin ) % if the minimum distance is above the threshold, add the current point to the list 
        iPick = [iPick, ii];
        count = count + 1;
    end

end

D = D(iPick,:); % keep only the 40000 points which are spread out from each other

dlmwrite('data.csv',D,'delimiter',',','precision','%.15e'); % output the data to data.csv