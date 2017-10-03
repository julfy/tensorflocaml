open Printf
open ExtLib
open Tensorflow_core
open Wrapper
module Graph = Wrapper.Graph
module Session = Wrapper.Session

let rex = Re2.Std.Re2.create_exn ~options:[`Case_sensitive false; `Dot_nl true; `Encoding_latin1 true] "(<br />|\n|[^a-zA-Z]+)"
let clean str = String.lowercase_ascii str |> Re2.Std.Re2.replace_exn ~f:(fun _ -> " ") rex
let t = Hashtbl.create 100000
let vocab = Hashtbl.create 100000

let shuffle d = (* switch to array? *)
    let nd = List.map (fun c -> (Random.bits (), c)) d in
    let sond = List.sort ~cmp:(fun (x,_) (y,_) -> compare x y) nd in
    List.map snd sond

let zero_pad n l =
  match max 0 (n - List.length l) with
  | 0 -> ExtLib.List.take n l
  | n -> let z = ExtLib.List.make n 0. in l @ z

let preprocess_str str =
  let words = String.nsplit (clean str) " " in
  List.filter_map begin function (* TODO: more interesting stuff can be done *)
  | "" -> None
  | word -> Hashtbl.find_opt vocab word
  end words

let init_str str =
  let words = String.nsplit (clean str) " " in
  List.iter begin function
  | "" -> ()
  | word ->
    match Hashtbl.find_opt t word with
    | None -> Hashtbl.add t word (ref 1)
    | Some n -> incr n
  end words

let init_preprocessor dir =
  let files = Sys.readdir dir in
  Array.iter begin fun fname ->
    init_str (ExtLib.input_file (dir ^ "/" ^ fname))
  end files

let filter_vocab () =
  let i = ref 1 in
  Hashtbl.iter (fun k v -> if !v > 5 then (Hashtbl.add vocab k (float_of_int !i); incr i)) t

let preprocess dir =
  let files = Sys.readdir dir in
  let rows = ref [] in
  Array.iter begin fun fname ->
    try
    let content = preprocess_str (ExtLib.input_file (dir ^ "/" ^ fname)) in
    let score = String.split fname "_" |> snd |> String.split_on_char '.' |> List.hd |> int_of_string in
    rows := (content, (if score > 5 then 1. else 0.))::!rows
    with _ -> ()
  end files;
  !rows

let load_data from_dirs =
  List.iter init_preprocessor from_dirs;
  filter_vocab ();
  List.fold_left (fun acc dir -> (preprocess dir) @ acc) [] from_dirs

let dump_preprocessed l =
  List.iter (fun (data,label) -> printf "%s,%.1f\n" (String.join "," (List.map string_of_float data)) label) (shuffle l)

let save_preprocess_params where =
  let out = open_out where in
  let oc = IO.output_channel out in
  Hashtbl.iter (fun word idx -> IO.write_line oc (sprintf "%s,%.0f" word idx)) vocab;
  close_out out

let load_vocabulary fname =
  let chan = open_in fname in
  List.iter (fun l -> let w,i = String.split l "," in Hashtbl.add vocab w (float_of_string i)) (Std.input_list chan);
  close_in chan

let gen_training_data () =
  let l = load_data ["data/aclImdb/train/neg";"data/aclImdb/train/pos";"data/aclImdb/test/neg";"data/aclImdb/test/pos"] in
  save_preprocess_params "vocab.csv";
  dump_preprocessed l;
  ()

let ok_exn (result : 'a Status.result) ~context =
  match result with
  | Ok result -> result
  | Error status ->
    Printf.sprintf "Error in %s: %s" context (Status.message status)
    |> failwith

let predict input =
  let make_interface_node graph name =
    match Graph.find_operation graph name with
    | Some op -> Graph.create_output op ~index:0
    | _ -> failwith (sprintf "no such node: %S" name)
  in
  load_vocabulary "vocab.csv";
  let input = ExtLib.input_file input in
  let tensor = Tensor.create2 Float32 1 1500 in
  Tensor.copy_elt_list tensor (input |> preprocess_str |> zero_pad 1500);
  let input_tensor = Tensor.P tensor in

  let graph = Graph.create () in
  Graph.import graph (Protobuf.read_file "models/final_.pb" |> Protobuf.to_string) |> ok_exn ~context:"import";

  let make_predictor graph inputs outputs =
    let session = Session.create graph |> ok_exn ~context:"create session" in
    let inps = List.map (make_interface_node graph) inputs in
    let outs = List.map (make_interface_node graph) outputs in
    (fun tensors ->
      Session.run session ~inputs:(List.combine inps tensors) ~outputs:outs |> ok_exn ~context:"run")
  in
  let predict = make_predictor graph ["input_1"] ["dense_1/Sigmoid"] in
  begin
    match predict [input_tensor] with
    | [ output ] -> printf "Ans: [%s]\n" (String.join " " (List.map string_of_float Tensor.(to_float_list output)))
    | _ -> assert false
  end
  (* del tensor, del output ? *)

let () =
  ExtArg.(parse [
    "-predict", String predict, " predict mood of input file 0-1 (models/final_.pb and it's interface nodes;vocab.csv are hardcoded)";
    "-gen-data", Unit gen_training_data, " generate vocabulary and training data (data/aclImdb/train/neg;data/aclImdb/train/pos;data/aclImdb/test/neg;data/aclImdb/test/pos;vocab.csv are hardcoded)"
  ])
