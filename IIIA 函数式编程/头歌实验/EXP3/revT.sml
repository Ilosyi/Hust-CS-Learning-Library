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

fun split [ ]  = ([ ], [ ]) 
    | split [x] = ([ ], [x])
    | split (x::y::L) =
	let val (A, B) =split L
	in (x::A, y::B) 	
	end;

datatype tree = Empty | Br of tree * int * tree; 

fun trav(Br(t1,a,t2)) = trav(t1)@(a::trav(t2))
    |trav empty = [];

(*BEGIN*)
fun listToTree ([] : int list) : tree = Empty
  | listToTree (x::l) = 
    let
      (* 1. 将列表的其余部分 l 分成两半 *)
      val (leftList, rightList) = split l
      
      (* 2. 递归地用这两半列表构建左右子树 *)
      val leftTree = listToTree(leftList)
      val rightTree = listToTree(rightList)
    in
      (* 3. 用第一个元素 x 作为根，组合成新树 *)
      Br(leftTree, x, rightTree)
    end

fun revT (Empty : tree) : tree = Empty
  | revT (Br(t1, a, t2)) = Br(revT(t2), a, revT(t1));
(*END*)

val L = getIntList(7);
printIntList (trav(revT(listToTree L))); 
