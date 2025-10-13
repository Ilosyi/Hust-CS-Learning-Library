fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
    
fun printIntList ( [] ) = ()
  | printIntList ( x::xs ) = 
    let
	val tmp = printInt(x)
    in
	printIntList(xs)
    end;

fun getIntList ( 0 ) = []
  | getIntList ( N:int) = getInt()::getIntList(N-1);

 
(*begin*)
fun findOdd ([]) : int option = NONE
  | findOdd (x::xs) = 
    if x mod 2 = 1 orelse x mod 2 = ~1 then 
        SOME x
    else 
        findOdd xs;
(*end*)


val INPUT = getIntList(6);
printInt (valOf(findOdd(INPUT)));
