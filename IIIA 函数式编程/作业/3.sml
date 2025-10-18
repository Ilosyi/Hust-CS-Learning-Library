fun subsetSumOption ([], s : int) : int list option = 
    if s = 0 then SOME []
    else NONE
  | subsetSumOption (x::xs, s) =
    case subsetSumOption(xs, s - x) of
        SOME subset => SOME (x::subset)  (* 包含x的情况 *)
      | NONE => subsetSumOption(xs, s);   (* 不包含x的情况 *)

subsetSumOption([1, 2, 3, 4], 5);  (* => SOME [2, 3] 或 SOME [1, 4] *)
subsetSumOption([1, 2, 3], 7);     (* => NONE *)
subsetSumOption([5, 10, 15], 15);  (* => SOME [15] *)
subsetSumOption([1, 2, 3], 0);     (* => SOME [] *)
