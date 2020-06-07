# this is going to be an attempt estimating Rt from the real-time data
# this version is going to try and optimize sigma as well. here we go:
# also has the code that calculates the HDIs.

using Plots
using DelimitedFiles
using Dates
using Distributions
using ImageFiltering

include("/Users/niqbalold/Dropbox/covid/simulator/dataAnalysis.jl");

println("Beginning")

# okay, I need a function to return the means and the deviations
# given an interval
function hdi(vals,prob,p)
    # given a probability distribution, returns the HDI and the maximum likelihood
    maxLike = argmax(prob);

    # now the HDI: the idea here is, find the smallest interval that contains a fraction
    # p of the probability mass

    N = length(vals);
    massContained = zeros(N,N);
    currMin = N;
    currLow = 0;
    currHigh = 0;
    for i = 1:N
        for j = i:N
            massContained[i,j] = sum(prob[i:j]);
            if (massContained[i,j] > p)
                # okay, this interval has enough weight
                if ((j - i) < currMin)
                    # then this is a smaller interval
                    currLow = i;
                    currHigh = j;
                    currMin = (j-i);
                end
            end

        end
    end



    return (vals[currLow], vals[currHigh], vals[maxLike]);
end



function meanDev(Rs, prob)
    mean = 0
    dev = 0;
    for i = 1:length(prob)
        mean+=Rs[i]*prob[i];
    end
    for i = 1:length(prob)
        dev += (Rs[i]-mean)^2*prob[i]
    end
    return (mean, sqrt(dev));
end


# load in the data

country = "Bangladesh";

(dates,deaths,raw_cases) = readData(country);

usingDeaths = false;

# try using the deaths instead:
if (usingDeaths)
    raw_cases = deaths;
end

# use only once the cases are greater than 25

# okay, its time to figure out the number of new cases every day.


deltaCasesRaw = raw_cases[2:end]-raw_cases[1:end-1];
minIndex = findall(x->x>=25,deltaCasesRaw)[1];
deltaCasesRaw = deltaCasesRaw[minIndex:end];
dates = dates[minIndex:end];
#=
data = readdlm("testing-increasing.txt",',');
deltaCasesRaw = data[:,1];
=#
# first, smoothen it using a gaussian filter
filterWidth = 3.5;



# apply the filter on the delta cases (removes the edge effects)
ker = ImageFiltering.Kernel.gaussian((filterWidth,));
deltaCases = floor.(imfilter(deltaCasesRaw,ker));
#cases = raw_cases;

# make a fake data set
#=cases = zeros(100);
for i = 1:100
    cases[i] = floor(200*min(exp(0.25*i),5000));
end =#
# okay now let's see: try to do my own implementation of this. following systrom's
# rt.live website, as well as Bettencourt and Ribeiro (real-time Bayesian estimation)


# here we go. store the probability distribution for R as a function of time.

days = length(deltaCases); # of days over which to run.
N = 201; # this is the graininess
minR = 0;
maxR = 3;
tR = 7; # serial interval in days; called gamma in the ref.


# the actual R values
Rs = range(minR, stop = maxR, length=N);

# we will also loop over sigmas
Nsigmas = 21;
sigmas = range(0,stop=0.5,length=Nsigmas);
logprobs = zeros(Nsigmas);

means = zeros(Nsigmas,days);
devs = zeros(Nsigmas,days);

# also plot the lows, the highs, and the max likelihoods
lows = zeros(Nsigmas, days);
highs = zeros(Nsigmas, days);
maxLikes = zeros(Nsigmas, days);

# don't store the full probability distribution for all sigmas
prob = zeros(days,N);

# initialize the first distribution (unknown prior)
prob[1,:] = ones(N)/N;
#prob[2,:] = ones(N)/N;

# this is where I store the *denominators* of Bayes rule
# in order to optimize the hyperparameter sigma.

logprob = 0;

# okay, here we go: I will now also consider the varial serial interval
# idea:

p_serial = Gamma(6.5,0.62)
# these figures are taken from
# page 18 of https://www.imperial.ac.uk/media/imperial-college/medicine/mrc-gida/2020-03-30-COVID19-Report-13.pdf

# now, need to construct a discrete serial interval from here; do it over the full length of days

ws = zeros(days);
ws[1] = cdf(p_serial,1.5);
for i = 2:days
    ws[i] = cdf(p_serial,i+0.5)-cdf(p_serial,i-0.5);
end

#error();

for s =1:length(sigmas)

    for day=2:days

        deltaIprev = deltaCases[day-1];
        deltaI = deltaCases[day];
        # add the gaussian noise to the old probability:
        # this is crucial to actually allow the R0 to drop
        # below 1; why is that? some kind of memory effect.
        # needs to be understood.

        # the determination of this hyperparameter is confusing.
        # the current value
        ker = ImageFiltering.Kernel.gaussian((sigmas[s]*N,));
        noised_p = imfilter(prob[day-1,:],ker);
        #noised_p = prob[day-1,:];
        for j=1:N
            # this max is to prevent a complete breakdown if ever the number of
            # new cases is zero.

            # now here is the thing: we will now use the serial interval distribution to compute this
            serial_weighted_infection = 0;
            for k = 1:(day-1)
                serial_weighted_infection += deltaCases[k]*ws[day-k]
            end
            #lambda = Rs[j]*serial_weighted_infection;
            #lambda = Rs[j]*deltaIprev;
            lambda = exp((Rs[j]-1)/tR)*(max(deltaIprev,1));
            #lambda = (1 + (Rs[j]-1)/tR)*(max(deltaIprev,1));
            # this is the expected value from the exponential growth.
            # make a distribution with this mean:
            p = Poisson(lambda);
            # assign the new value.

            # add the gaussia noise to the old one:

            #prob[day,j] = pdf.(p,deltaI)*prob[day-1,j];
            prob[day,j] = abs(pdf.(p,deltaI)*noised_p[j]);
        end
        # normalize it:
        norm = sum(prob[day,:]);

        prob[day,:] = prob[day,:]/norm;
        logprobs[s] += log(norm);
        #deltaIprev = deltaI;
        (means[s,day],devs[s,day])=meanDev(Rs,prob[day,:]);

        # display the 0.9 HDI
        (lows[s,day], highs[s,day], maxLikes[s,day]) = hdi(Rs,prob[day,:],0.9);
    end
end

s = argmax(logprobs);
#display(plot(means[s,3:end],yerrors=devs[s,3:end],xlabel=string("Days since ",dates[3], ", last day is ", dates[end], " final R ", means[s,end]),title=string("Estimation of R for ",country, (usingDeaths ? " using Deaths" : " using Reported Cases")),ylabel=string("R (sigma = ", sigmas[s],")"),legend=false));

display(plot([lows[s,3:end] highs[s,3:end] maxLikes[s,3:end]],xlabel=string("Days since ",dates[3], ", last day is ", dates[end], " final R ", maxLikes[s,end]),title=string("Estimation of R for ",country, (usingDeaths ? " using Deaths" : " using Reported Cases")),ylabel=string("R (sigma = ", sigmas[s],")"),legend=false));
#display(plot(means[3:end],yerrors=devs[3:end],ylabel="R",legend=false));
# now make a plot of the peak of the distributioin:
display(hline!([1]));
savefig(string("output files/",now(),"-Rs-old-algorithm.png"));
println("Optimum value of sigma: ", sigmas[s]);
writedlm(string("output files/",now(),"-Rs-old-algorithm.txt"),[means devs]);

# okay, now try something different (as a null case)
#=newCasesRaw = raw_cases[2:end]-raw_cases[1:end-1];
newCasesSmooth = cases[2:end]-cases[1:end-1];
fracChange = log.(newCasesSmooth[2:end]./newCasesSmooth[1:end-1]);
R_simplest = (fracChange)*tR+ones(length(fracChange));
=#
#display(plot(R_simplest));
#display(plot(newCasesSmooth));
#savefig("R_deathsBangladesh.pdf");

#plot(fracChange);
