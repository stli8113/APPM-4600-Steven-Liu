clear all

[q,fcnEvals] = quad(@f, 0, 50)


function gamma = f(x)
    gamma = x.^(10-1) .* exp(-x);
end