(* ================= 工具函数部分 ================= *)

fun printInt (a : int) =
    print (Int.toString a ^ " ");

fun getInt () =
    Option.valOf (TextIO.scanStream (Int.scan StringCvt.DEC) TextIO.stdIn);

fun printIntList [] = ()
  | printIntList (x :: xs) =
      let
          val _ = printInt x
      in
          printIntList xs
      end;

fun getIntList 0 = []
  | getIntList (n : int) = getInt() :: getIntList (n - 1);

(* ================= 列表工具函数 ================= *)

fun split [] = ([], [])
  | split [x] = ([], [x])
  | split (x :: y :: L) =
      let
          val (A, B) = split L
      in
          (x :: A, y :: B)
      end;

(* ================= 二叉树与遍历 ================= *)

datatype tree = Empty | Br of tree * int * tree;


fun trav (Br (t1, a, t2)) = trav t1 @ (a :: trav t2)
  | trav Empty = [];

fun listToTree ([] : int list) : tree = Empty
  | listToTree (x :: l) =
      let
          val (l1, l2) = split l
      in
          Br (listToTree l1, x, listToTree l2)
      end;

(* ================= 树的比较与堆化 ================= *)

fun treecompare (Br (l1, x, r1), Br (l2, y, r2)) = Int.compare (x, y)
  | treecompare (Empty, Br(t1,x,t2)) = LESS
  | treecompare (Br(t1,x,t2), Empty) = GREATER
  | treecompare (Empty, Empty) = EQUAL;

(* SwapDown：堆的下滤操作 *)
fun SwapDown Empty = Empty
  | SwapDown (Br (Empty, v, Empty)) = Br (Empty, v, Empty)
  | SwapDown (Br (Empty, v, Br (l2, y, r2))) =
      if v > y then Br (Empty, y, SwapDown (Br (l2, v, r2)))
      else Br (Empty, v, Br (l2, y, r2))
  | SwapDown (Br (Br (l1, x, r1), v, Empty)) =
      if v > x then Br (SwapDown (Br (l1, v, r1)), x, Empty)
      else Br (Br (l1, x, r1), v, Empty)
  | SwapDown (Br (Br (l1, x, r1), v, Br (l2, y, r2))) =
      if y < x andalso v > y then
        Br (Br (l1, x, r1), y, SwapDown (Br (l2, v, r2)))
      else if x <= y andalso v > x then
        Br (SwapDown (Br (l1, v, r1)), x, Br (l2, y, r2))  
      else
        Br (Br (l1, x, r1), v, Br (l2, y, r2));

(* heapify：将一棵任意二叉树转为最小堆 *)
fun heapify Empty = Empty
  | heapify (Br (l, x, r)) =
      let
          val heapL = heapify l
          val heapR = heapify r
      in
          SwapDown (Br (heapL, x, heapR))
      end;

(* ================= 主程序入口 ================= *)

val L = getIntList(7);
printIntList (trav(heapify(listToTree L)));
