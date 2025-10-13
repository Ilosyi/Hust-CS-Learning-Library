 fun printBool (a:bool) = 
    print(Bool.toString(a)^" "); 

 fun printInt (a:int) =
    print(Int.toString(a)^" ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);
 
(*begin*)
fun thenAddOne (f : int -> int, n : int) : int = 
    (f n) + 1;
(*end*)

fun double (0 : int) : int = 0
    | double n = 2 + double (n - 1);

val OUT = thenAddOne (double, 3);

printInt(OUT);
