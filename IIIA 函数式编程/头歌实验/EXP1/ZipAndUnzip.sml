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

 
(*** Begin ***)
 (* zip : string list * int list -> (string * int) list *)
 fun zip ([], _) = []
   | zip (_, []) = []
   | zip (s::ss, i::ii) = (s, i) :: zip(ss, ii);

 (* unzip : (string * int) list -> string list * int list *)
  fun unzip ([]) = ([], [])
    | unzip ((s, i)::xs) =
        let
            val (ss, ii) = unzip(xs)
        in
            (s::ss, i::ii)
        end;
(*** End ***)

val test1 = [("a",1), ("b",2)] = zip(["a","b"],[1,2]);
print (Bool.toString test1 ^ "\n");

val test2 = [("a",1)] = zip(["a"],[1,2,3]);
print (Bool.toString test2 ^ "\n");

val test3 = [("a",1), ("b",2)] = zip(["a","b","c","d"],[1,2]);
print (Bool.toString test3 ^ "\n");

val test4 = (["a","b"],[1,2]) = unzip([("a",1), ("b",2)]);
print (Bool.toString test4 ^ "\n");

val test5 = (["dragon","llama","muffin"],[42,54,76]) =
              unzip([("dragon",42),("llama",54),("muffin",76)]);
print (Bool.toString test5 ^ "\n");
