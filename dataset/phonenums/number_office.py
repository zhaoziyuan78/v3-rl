import os
import argparse
from tqdm import tqdm
from random import shuffle

from dataset.phonenums.typography import Typography


class NumberOffice:
    def __init__(
        self,
        pagesize="20x1",
        patchsize="32x48",
        font="./dataset/phonenums/fonts/ITCKRIST.TTF",
        output_dir=None,
    ) -> None:
        # load typography
        self.typography = Typography(pagesize=pagesize, patchsize=patchsize)
        self.font = font

        self.output_dir = output_dir

    def generate_folder(self, n_pages=100000):
        """
        pallete for training: ["black", "blue", "green", "red", "teal", "purple", "orange", "brown"]
        pallete for OOD: ["pink", "salmon", "gold", "lime", "cyan", "magenta", "gray", "peru"]
        """
        from matplotlib import colors

        os.makedirs(self.output_dir, exist_ok=True)
        pallete = [
            "black",
            "blue",
            "green",
            "red",
            "teal",
            "purple",
            "orange",
            "brown",
        ]

        for i in tqdm(range(n_pages)):
            # challenging mode
            # numbers = np.random.randint(0, 10, self.typography.page_width)
            # standard mode
            numbers = list(range(10)) * (self.typography.page_width // 10)

            shuffle(numbers)
            digits = "".join([str(n) for n in numbers])

            color = list(colors.to_rgb(pallete[i % len(pallete)]))
            color = tuple([int(c * 240 + 8) for c in color])
            output_name = f"{digits}_{pallete[i % len(pallete)]}.png"
            output_path = os.path.join(self.output_dir, output_name)
            self.typography.printer(
                digits,
                font=self.font,
                fg=color,
                output_path=output_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate phone number images")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/phonenums_",
        help="Directory to save the generated phone number images",
    )
    parser.add_argument(
        "--n_pages",
        type=int,
        default=1000,
        help="Number of pages to generate",
    )
    args = parser.parse_args()

    press = NumberOffice(
        pagesize="10x1",
        patchsize="32x48",
        output_dir=args.save_dir,
    )
    press.generate_folder(n_pages=args.n_pages)
    print(f"Generated {args.n_pages} pages of phone number images")
    print(f"Saved to {args.save_dir}")
