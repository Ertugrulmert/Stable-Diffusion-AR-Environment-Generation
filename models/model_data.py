import numpy as np


class ModelData:

    CONTROLNET_MODEL_IDS = {
        'canny': 'lllyasviel/sd-controlnet-canny',
        'hough': 'lllyasviel/sd-controlnet-mlsd',
        'hed': 'lllyasviel/sd-controlnet-hed',
        'scribble': 'lllyasviel/sd-controlnet-scribble',
        'pose': 'lllyasviel/sd-controlnet-openpose',
        'seg': 'lllyasviel/sd-controlnet-seg',
        'depth': 'lllyasviel/sd-controlnet-depth',
        'normal': 'lllyasviel/sd-controlnet-normal',
    }


    PROMPT_LIST = [
        # ---- Prompts taken from the "SceneScape: Text-Driven Consistent Scene Generation" paper

        "a dimly lit library, with rows upon rows of leather-bound books and dark wooden shelves",
        "A grand, marble staircase spirals up to a vaulted ceiling in a grand entrance hall of a palace.",
        "POV, haunted house, dark, wooden door, spider webs, skeletons",
        "indoor scene, interior, candy house, fantasy, beautiful, masterpiece, best quality",
        "inside a castle made of ice, beautiful photo, masterpiece",
        "walkthrough, inside a medieval forge, metal, fire, beautiful photo, masterpiece",
        "walkthrough, sci-fi ship interiors, corridors,amazing quality, masterpiece, beautiful scenery, best quality",
        "POV, cave, pools, water, dark cavern, inside a cave, beautiful scenery, best quality",

        # ---- Prompts taken from the "Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models" paper
        "Editorial Style Photo, Coastal Bathroom, Clawfoot Tub, Seashell, Wicker, Mosaic Tile, Blue and White"
        "A living room with a lit furnace, couch, and cozy curtains, bright lamps that make the room look well-lit",
        "Editorial Style Photo, Modern Living Room, Large Window, Leather, Glass, Metal, Wood Paneling, Apartment",
        "Editorial Style Photo, Modern Nursery, Table Lamp, Rocking Chair, Tree Wall Decal, Wood, Cotton, Faux Fur",
        "Editorial Style Photo, Industrial Home Office, Steel Shelves, Concrete, Metal, Edison Bulbs, Exposed Ductwork",
        "a bedroom with a king-size bed and a large wardrobe",
    ]


    interior_design_prompt_1 = "Intricate, Ornate, Embellished, Elaborate, Detailed, Decorative, Intricately-crafted, Luxurious, Ornamented, and Artistic cloak, open book, sparks, cozy library in background, furniture, fire place, food, wine, pet, chandelier, High Definition, Night time, Photorealism, realistic"
    interior_design_prompt_2 = "Residential home high end futuristic interior, olson kundig, Interior Design by Dorothy Draper, maison de verre, axel vervoordt, award winning photography of an indoor-outdoor living library space, minimalist modern designs, high end indoor/outdoor residential living space, rendered in vray, rendered in octane, rendered in unreal engine, architectural photography, photorealism, featured in dezeen, cristobal palma. 5 chaparral landscape outside, black surfaces/textures for furnishings in outdoor space"

    additional_prompt = ', a detailed high-quality professional image'
    negative_prompt = ', lowres, cropped, worst quality, low quality'

    """ADE20K palette that maps each class to RGB values."""
    ade_palette = np.array([[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                [102, 255, 0], [92, 0, 255]])