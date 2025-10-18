fun hd (x::_) = x;
fun next(xlist, y::ys) : int list =
              if hd xlist < y 
              then next(y::xlist, ys)
              else   let fun swap [x] = y::x::ys
                                |  swap (x::xk::xs)  = if xk > y
                                         then x::swap(xk::xs)
                                        else (y::xk::xs)@(x::xs)
                        in swap (xlist)
                        end;
fun nextperm(y::ys) = next ([y], ys);

nextperm[2,3,1,4];